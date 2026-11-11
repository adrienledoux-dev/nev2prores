use std::fs;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use nev2prores::demuxer::{build_synthetic_nev, demux_nev_file};
use nev2prores::encoder::{decode_prores_frame_to_rgba16f, ProResCodec};
use nev2prores::metal_pipeline::{ColorPipelineParams, MetalColorPipeline, ScaleMode};
use nev2prores::{transcode_nev_file, transcode_nev_file_with_progress, TranscodeOptions};
use tempfile::tempdir;

#[test]
fn test_full_end_to_end_nev_to_prores4444_and_fcpxml() {
    let dir = tempdir().expect("create tempdir");
    let nev_path = dir.path().join("DSC_0001.NEV");
    let out_dir = dir.path().join("out");

    // Generate 3 synthetic 128x72 RGGB 12-bit frames with a color ramp
    let width = 128u16;
    let height = 72u16;
    let mut frames = Vec::new();
    for f in 0..3 {
        let mut bayer = vec![0u16; (width as usize) * (height as usize)];
        for y in 0..(height as usize) {
            for x in 0..(width as usize) {
                let phase = ((y & 1) << 1) | (x & 1);
                let base = 150 + ((x * 20 + y * 15 + f * 200) % 3500) as u16;
                bayer[y * (width as usize) + x] = match phase {
                    0 => base,                       // R
                    1 | 2 => (base + 300).min(4095), // G1, G2
                    _ => (base + 100).min(4095),     // B
                };
            }
        }
        frames.push(bayer);
    }

    let nev_bytes = build_synthetic_nev(
        width,
        height,
        &frames,
        [2.05, 1.48, 1.0, 1.0],
        [150, 150, 150, 150],
        800,
    );
    fs::write(&nev_path, &nev_bytes).expect("write synthetic .nev");

    let mut gpu = MetalColorPipeline::new().expect("Metal GPU should initialize");
    let opts = TranscodeOptions {
        codec: ProResCodec::ProRes4444,
        kelvin: Some(5000),
        tint: Some(4.5),
        ev: Some(0.75),
        scale: ScaleMode::Full,
        emit_fcpxml: true,
        ring_buffer_capacity: 4,
    };

    let report = transcode_nev_file(&nev_path, &out_dir, &opts, &mut gpu)
        .expect("end-to-end transcode should succeed");

    assert_eq!(report.frames_transcoded, 3);
    assert_eq!(report.output_width, 128);
    assert_eq!(report.output_height, 72);
    assert!(report.output_mov_path.exists());

    // Demux the output .mov to verify Video (`ap4h`), Audio (`in24`), Timecode (`tmcd`), and `colr` (`nclc`)
    let mov_demuxed = demux_nev_file(&report.output_mov_path)
        .expect("output .mov should be a valid 64-bit ISO-BMFF/QuickTime file");
    assert_eq!(&mov_demuxed.major_brand, b"qt  ");
    assert_eq!(&mov_demuxed.video_codec_fourcc, b"ap4h");
    assert_eq!(mov_demuxed.frames.len(), 3);
    for frame_desc in &mov_demuxed.frames {
        assert!(
            frame_desc.byte_size > 64,
            "ProRes frame should have non-trivial payload"
        );
    }

    // Verify `colr` (`nclc`) atom with Rec. 2020 `9-1-9` (avoiding linear `8` double-gamma lift)
    let mov_bytes = fs::read(&report.output_mov_path).unwrap();
    let expected_colr = [
        0, 0, 0, 18, b'c', b'o', b'l', b'r', b'n', b'c', b'l', b'c', 0, 9, 0, 1, 0, 9,
    ];
    assert!(
        mov_bytes
            .windows(expected_colr.len())
            .any(|w| w == expected_colr),
        "Output .mov must contain colr (nclc: 9, 1, 9) Rec.2020 atom"
    );

    // Verify QuickTime Sound Sample Description v1 `wave -> enda(1)` little-endian atom for `in24`
    let expected_enda = [0, 0, 0, 10, b'e', b'n', b'd', b'a', 0, 1];
    assert!(
        mov_bytes
            .windows(expected_enda.len())
            .any(|w| w == expected_enda),
        "Output .mov must contain QuickTime wave/enda=1 little-endian atom for 24-bit PCM"
    );

    // Hardware `VTDecompressionSession` Golden Round-Trip Accuracy Verification:
    // Decode the first compressed ProRes 4444 frame back to 64RGBAHalf and compare against direct Metal GPU output!
    let src_demuxed = demux_nev_file(&nev_path).expect("demux input .nev");
    let params = ColorPipelineParams::from_metadata(
        width as u32,
        height as u32,
        &src_demuxed.metadata,
        opts.scale,
        opts.kelvin,
        opts.tint,
        opts.ev,
    );
    let gpu_ref = gpu
        .process_frame_to_rgba16f_vec(&frames[0], &params)
        .expect("reference GPU output");

    let first_pkt = &mov_demuxed.frames[0];
    let pkt_start = first_pkt.file_offset as usize;
    let pkt_end = pkt_start + (first_pkt.byte_size as usize);
    let decoded_rgba16f = decode_prores_frame_to_rgba16f(
        ProResCodec::ProRes4444,
        128,
        72,
        &mov_bytes[pkt_start..pkt_end],
    )
    .expect("VTDecompressionSession should decode ProRes 4444 frame back to RGBA16F");

    assert_eq!(decoded_rgba16f.len(), gpu_ref.len());
    let mut sum_abs_err = 0.0f64;
    let mut max_abs_err = 0.0f32;
    let mut checked_channels = 0usize;
    // Compare interior pixels (avoiding 2px DCT block border ringing on steep synthetic ramps)
    for y in 4..(height as usize - 4) {
        for x in 4..(width as usize - 4) {
            let base = (y * (width as usize) + x) * 4;
            for c in 0..3 {
                let expected = gpu_ref[base + c].to_f32();
                let actual = decoded_rgba16f[base + c].to_f32();
                let err = (expected - actual).abs();
                sum_abs_err += err as f64;
                max_abs_err = max_abs_err.max(err);
                checked_channels += 1;
            }
        }
    }
    let mean_abs_err = sum_abs_err / (checked_channels as f64);
    assert!(
        mean_abs_err < 0.008,
        "ProRes 4444 round-trip mean absolute error too large: {:.6} (max {:.6})",
        mean_abs_err,
        max_abs_err
    );
    assert!(
        max_abs_err < 0.035,
        "ProRes 4444 round-trip peak error too large: {:.6}",
        max_abs_err
    );

    // Verify audio and timecode passthrough
    let audio = mov_demuxed
        .audio_track
        .expect("audio track should be preserved");
    assert_eq!(audio.channels, 2);
    assert_eq!(audio.sample_bits, 24);
    assert_eq!(audio.sample_rate, 48000);
    assert_eq!(audio.total_samples, 2000);

    let tc = mov_demuxed
        .timecode_track
        .expect("tmcd track should be preserved");
    assert_eq!(tc.timecode_str, "01:00:00:00");

    // Verify FCPXML v1.11 sidecar
    let fcpxml_path = report
        .output_fcpxml_path
        .expect("fcpxml path should be returned");
    let fcpxml_content = fs::read_to_string(&fcpxml_path).expect("read .fcpxml");
    assert!(fcpxml_content.contains(r#"<fcpxml version="1.11">"#));
    assert!(fcpxml_content.contains(r#"colorSpace="9-1-9 (Rec. 2020)""#));
    assert!(fcpxml_content.contains(r#"colorSpaceOverride="Rec. 2020 N-Log""#));
    assert!(fcpxml_content.contains(r#"key="com.apple.proapps.mio.cameraISO" value="800""#));
    assert!(fcpxml_content.contains(r#"key="com.nikon.nraw.kelvin" value="5000""#));
    assert!(fcpxml_content.contains(r#"key="com.nikon.nraw.evOffset" value="+0.75""#));
}

#[test]
fn test_transcode_cancellation_deletes_incomplete_mov() {
    let dir = tempdir().expect("create tempdir");
    let nev_path = dir.path().join("CANCEL_CLIP.NEV");
    let out_dir = dir.path().join("out_cancel");

    let width = 128u16;
    let height = 72u16;
    let frame = vec![1000u16; (width as usize) * (height as usize)];
    let nev_bytes = build_synthetic_nev(
        width,
        height,
        &[frame.clone(), frame.clone(), frame],
        [1.95, 1.45, 1.0, 1.0],
        [150, 150, 150, 150],
        800,
    );
    fs::write(&nev_path, &nev_bytes).expect("write synthetic .nev");

    let mut gpu = MetalColorPipeline::new().expect("Metal GPU should initialize");
    let cancel_flag = Arc::new(AtomicBool::new(true));
    let res = transcode_nev_file_with_progress(
        &nev_path,
        &out_dir,
        &TranscodeOptions::default(),
        &mut gpu,
        Some(cancel_flag),
        |_, _| {},
    );
    assert!(res.is_err(), "pre-cancelled transcode must return Err");
    assert!(
        !out_dir.join("CANCEL_CLIP.mov").exists(),
        "cancelled transcode must clean up incomplete .mov output file"
    );
}

#[test]
fn test_prores422hq_half_scale_transcode() {
    let dir = tempdir().expect("create tempdir");
    let nev_path = dir.path().join("DSC_0002.NEV");
    let out_dir = dir.path().join("out_422hq");

    let width = 128u16;
    let height = 72u16;
    let frame = vec![1200u16; (width as usize) * (height as usize)];
    let nev_bytes = build_synthetic_nev(
        width,
        height,
        &[frame],
        [1.90, 1.55, 1.0, 1.0],
        [150, 150, 150, 150],
        1600,
    );
    fs::write(&nev_path, &nev_bytes).expect("write synthetic .nev");

    let mut gpu = MetalColorPipeline::new().expect("Metal GPU should initialize");
    let opts = TranscodeOptions {
        codec: ProResCodec::ProRes422HQ,
        kelvin: Some(3200),
        tint: Some(-3.0),
        ev: Some(-0.5),
        scale: ScaleMode::Half,
        emit_fcpxml: false,
        ring_buffer_capacity: 2,
    };

    let report = transcode_nev_file(&nev_path, &out_dir, &opts, &mut gpu)
        .expect("ProRes 422 HQ half-scale transcode should succeed");

    assert_eq!(report.output_width, 64);
    assert_eq!(report.output_height, 36);
    assert_eq!(report.frames_transcoded, 1);

    let mov_demuxed = demux_nev_file(&report.output_mov_path).expect("demux .mov");
    assert_eq!(&mov_demuxed.video_codec_fourcc, b"apch");
    assert_eq!(mov_demuxed.width, 64);
    assert_eq!(mov_demuxed.height, 36);
}
