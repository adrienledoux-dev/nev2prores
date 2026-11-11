pub mod decoder;
pub mod demuxer;
pub mod encoder;
pub mod fcpxml;
pub mod gui;
pub mod lut;
pub mod metadata;
pub mod metal_pipeline;

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};

use crate::decoder::{BoundedDecodePipeline, NativeTicoRawDecoder};
use crate::demuxer::{build_synthetic_nev, demux_nev_file, NevDemuxedContainer};
use crate::encoder::{ProResCodec, QuickTimeMovWriter, VideoToolboxProResEncoder};
use crate::fcpxml::write_fcpxml_sidecar;
use crate::metal_pipeline::{ColorPipelineParams, MetalColorPipeline, ScaleMode};

/// Complete configuration options for a transcode job.
#[derive(Debug, Clone)]
pub struct TranscodeOptions {
    pub codec: ProResCodec,
    pub kelvin: Option<u32>,
    pub tint: Option<f32>,
    pub ev: Option<f32>,
    pub scale: ScaleMode,
    pub emit_fcpxml: bool,
    pub ring_buffer_capacity: usize,
}

impl Default for TranscodeOptions {
    fn default() -> Self {
        Self {
            codec: ProResCodec::ProRes4444,
            kelvin: None,
            tint: None,
            ev: None,
            scale: ScaleMode::Full,
            emit_fcpxml: false,
            ring_buffer_capacity: 4,
        }
    }
}

/// Summary report returned after transcoding a single `.nev` file.
#[derive(Debug, Clone)]
pub struct TranscodeReport {
    pub input_path: PathBuf,
    pub output_mov_path: PathBuf,
    pub output_fcpxml_path: Option<PathBuf>,
    pub frames_transcoded: usize,
    pub output_width: u32,
    pub output_height: u32,
    pub elapsed_secs: f64,
}

/// Prints a detailed inspection report for a `.nev` container (`--inspect-only`).
pub fn print_inspection_report(
    path: &Path,
    container: &NevDemuxedContainer,
    opts: &TranscodeOptions,
) {
    let (_, _, fps) = container.frame_rate();
    let (out_w, _out_h) = opts
        .scale
        .output_dimensions(container.width, container.height);
    let params = ColorPipelineParams::from_metadata(
        container.width,
        container.height,
        &container.metadata,
        opts.scale,
        opts.kelvin,
        opts.tint,
        opts.ev,
    );
    let uniforms = params.build_uniforms(out_w);

    println!("================================================================");
    println!(
        "Nikon N-RAW (.nev) Container Inspection: {}",
        path.display()
    );
    println!("================================================================");
    println!(
        "  Container Brand : '{}'",
        String::from_utf8_lossy(&container.major_brand)
    );
    println!(
        "  Video Track     : '{}' ({}x{}, {}-bit Bayer RGGB, {:.3} fps, {} frames)",
        String::from_utf8_lossy(&container.video_codec_fourcc),
        container.width,
        container.height,
        container.bit_depth,
        fps,
        container.frames.len()
    );
    if let Some(ref audio) = container.audio_track {
        println!(
            "  Audio Track     : '{}' ({} ch, {}-bit Linear PCM @ {} Hz, {} samples)",
            String::from_utf8_lossy(&audio.codec_fourcc),
            audio.channels,
            audio.sample_bits,
            audio.sample_rate,
            audio.total_samples
        );
    } else {
        println!("  Audio Track     : None");
    }
    if let Some(ref tc) = container.timecode_track {
        println!(
            "  Timecode Track  : {} (frame_number={}, {} fps)",
            tc.timecode_str, tc.frame_number, tc.num_frames
        );
    } else {
        println!("  Timecode Track  : None");
    }
    println!("----------------------------------------------------------------");
    println!("  Nikon NCDT MakerNote Metadata:");
    println!(
        "    Camera Model        : {}",
        container.metadata.camera_model
    );
    println!(
        "    Lens Model          : {}",
        container.metadata.lens_model
    );
    println!("    ISO Speed           : {}", container.metadata.iso_speed);
    println!(
        "    Shutter Speed       : {}",
        container.metadata.shutter_display()
    );
    println!(
        "    As-Shot WB (0x000C) : R={:.4}, B={:.4}, G1={:.4}, G2={:.4}",
        container.metadata.wb_rb_levels[0],
        container.metadata.wb_rb_levels[1],
        container.metadata.wb_rb_levels[2],
        container.metadata.wb_rb_levels[3]
    );
    println!(
        "    BlackLevel (0x003D) : RGGB = {:?} (WhiteLevel = {})",
        container.metadata.black_level, container.metadata.white_level
    );
    println!(
        "    Color Temperature   : {} K (Target Override: {:?} K, Tint: {:?}, EV: {:+.2})",
        container.metadata.color_temperature_kelvin, opts.kelvin, opts.tint, params.ev_stops
    );
    println!("----------------------------------------------------------------");
    println!("  Bradford Chromatic Adaptation Matrix M_WB (Camera -> Rec. 2020):");
    println!(
        "    [ {:+.5}, {:+.5}, {:+.5} ]",
        uniforms.color_matrix_row0[0], uniforms.color_matrix_row0[1], uniforms.color_matrix_row0[2]
    );
    println!(
        "    [ {:+.5}, {:+.5}, {:+.5} ]",
        uniforms.color_matrix_row1[0], uniforms.color_matrix_row1[1], uniforms.color_matrix_row1[2]
    );
    println!(
        "    [ {:+.5}, {:+.5}, {:+.5} ]",
        uniforms.color_matrix_row2[0], uniforms.color_matrix_row2[1], uniforms.color_matrix_row2[2]
    );
    println!("================================================================");
}

/// Runs the full 4-stage native transcode pipeline on a single `.nev` file.
pub fn transcode_nev_file(
    input_nev: &Path,
    output_dir: &Path,
    opts: &TranscodeOptions,
    gpu: &mut MetalColorPipeline,
) -> Result<TranscodeReport> {
    transcode_nev_file_with_progress(input_nev, output_dir, opts, gpu, None, |_, _| {})
}

/// Runs the full 4-stage native transcode pipeline on a single `.nev` file, invoking
/// `on_progress(completed_frames, total_frames)` as each frame is hardware-encoded,
/// and checking `cancel_flag` so partial `.mov` files are deleted cleanly upon cancellation.
pub fn transcode_nev_file_with_progress<F>(
    input_nev: &Path,
    output_dir: &Path,
    opts: &TranscodeOptions,
    gpu: &mut MetalColorPipeline,
    cancel_flag: Option<Arc<AtomicBool>>,
    mut on_progress: F,
) -> Result<TranscodeReport>
where
    F: FnMut(usize, usize),
{
    let start_time = Instant::now();

    // Stage 1: Zero-Copy 64-Bit ISO-BMFF Demuxer & NCDT MakerNote Parser
    let container = demux_nev_file(input_nev)
        .with_context(|| format!("Failed to demux '{}'", input_nev.display()))?;
    let total_frames = container.frames.len();

    let (out_width, out_height) = opts
        .scale
        .output_dimensions(container.width, container.height);

    let params = ColorPipelineParams::from_metadata(
        container.width,
        container.height,
        &container.metadata,
        opts.scale,
        opts.kelvin,
        opts.tint,
        opts.ev,
    );

    std::fs::create_dir_all(output_dir).with_context(|| {
        format!(
            "Failed to create output directory '{}'",
            output_dir.display()
        )
    })?;

    let stem = input_nev
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let output_mov_path = output_dir.join(format!("{}.mov", stem));

    // Stage 2: Multithreaded TicoRAW Bitstream Decoder (bounded channel + rayon worker pool)
    let mut raw_decoder = NativeTicoRawDecoder::new();
    raw_decoder.default_black_level = container.metadata.black_level[0];

    let decode_pipeline = BoundedDecodePipeline::spawn(
        input_nev.to_path_buf(),
        container.frames.clone(),
        container.width as usize,
        container.height as usize,
        Arc::new(raw_decoder),
        opts.ring_buffer_capacity,
        cancel_flag.clone(),
    );

    // Stage 4: Hardware VideoToolbox ProRes Session + 64-Bit QuickTime `.mov` Writer
    let mut vt_encoder =
        VideoToolboxProResEncoder::new(out_width, out_height, container.timescale, opts.codec)
            .context("Failed to initialize Apple VideoToolbox ProRes hardware session")?;

    let mut mov_writer = QuickTimeMovWriter::create(
        &output_mov_path,
        out_width,
        out_height,
        container.timescale,
        opts.codec,
    )
    .with_context(|| {
        format!(
            "Failed to create output file '{}'",
            output_mov_path.display()
        )
    })?;

    // Stream frames: CPU TicoRAW decode -> Metal GPU Debayer & N-Log -> VideoToolbox ProRes -> .mov
    let mut frames_transcoded = 0usize;
    for decoded_res in decode_pipeline.receiver().iter() {
        if cancel_flag
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            drop(mov_writer);
            drop(vt_encoder);
            let _ = std::fs::remove_file(&output_mov_path);
            bail!("Transcode cancelled by user");
        }

        let decoded_frame = match decoded_res {
            Ok(frame) => frame,
            Err(e) => {
                drop(mov_writer);
                drop(vt_encoder);
                let _ = std::fs::remove_file(&output_mov_path);
                return Err(e).context("TicoRAW frame decode failed");
            }
        };

        let packets = match vt_encoder.encode_bayer_frame_with_metal(
            gpu,
            &decoded_frame.bayer_rggb,
            &params,
            decoded_frame.pts,
            decoded_frame.duration,
        ) {
            Ok(pkts) => pkts,
            Err(e) => {
                drop(mov_writer);
                drop(vt_encoder);
                let _ = std::fs::remove_file(&output_mov_path);
                return Err(e).context("Failed to encode frame via Metal + VideoToolbox");
            }
        };

        for pkt in &packets {
            mov_writer.write_video_packet(pkt)?;
        }
        frames_transcoded += 1;
        on_progress(frames_transcoded, total_frames);
    }

    // Flush remaining frames from VideoToolbox
    let trailing_packets = vt_encoder.finish()?;
    for pkt in &trailing_packets {
        mov_writer.write_video_packet(pkt)?;
    }

    // Passthrough uncompressed 24-bit Linear PCM audio track & camera timecode (`tmcd`)
    if container.audio_track.is_some() || container.timecode_track.is_some() {
        let mut src_reader = BufReader::new(File::open(input_nev)?);
        if let Some(ref audio) = container.audio_track {
            mov_writer.passthrough_audio_track(&mut src_reader, audio)?;
        }
        if let Some(ref tc) = container.timecode_track {
            mov_writer.passthrough_timecode_sample(tc)?;
        }
    }

    mov_writer.finalize(
        container.audio_track.as_ref(),
        container.timecode_track.as_ref(),
    )?;

    // Optional FCPXML v1.11 Sidecar generation (`--fcpxml`)
    let output_fcpxml_path = if opts.emit_fcpxml {
        let fcpxml_path = output_dir.join(format!("{}.fcpxml", stem));
        write_fcpxml_sidecar(
            &fcpxml_path,
            &output_mov_path,
            &container,
            &params,
            opts.codec,
        )
        .with_context(|| {
            format!(
                "Failed to write FCPXML v1.11 sidecar '{}'",
                fcpxml_path.display()
            )
        })?;
        Some(fcpxml_path)
    } else {
        None
    };

    Ok(TranscodeReport {
        input_path: input_nev.to_path_buf(),
        output_mov_path,
        output_fcpxml_path,
        frames_transcoded,
        output_width: out_width,
        output_height: out_height,
        elapsed_secs: start_time.elapsed().as_secs_f64(),
    })
}

/// Generates a synthetic 1920x1080 12-bit RGGB Nikon Z9 `.nev` test file with SMPTE color bars,
/// a scene-linear exposure ramp, 24-bit 48kHz stereo PCM audio, `tmcd` timecode, and `NCDT` MakerNote.
pub fn generate_sample_nev_file(path: &Path) -> Result<()> {
    let width = 1920u16;
    let height = 1080u16;
    let num_frames = 24usize;
    let wb_rb_levels = [1.95f32, 1.42f32, 1.0f32, 1.0f32];
    let black_level = [150u16, 150, 150, 150];

    let bars_rgb: [[f32; 3]; 7] = [
        [0.75, 0.75, 0.75], // 75% White
        [0.75, 0.75, 0.05], // Yellow
        [0.05, 0.75, 0.75], // Cyan
        [0.05, 0.75, 0.05], // Green
        [0.75, 0.05, 0.75], // Magenta
        [0.75, 0.05, 0.05], // Red
        [0.05, 0.05, 0.75], // Blue
    ];

    let mut frames = Vec::with_capacity(num_frames);
    for f in 0..num_frames {
        let mut bayer = vec![0u16; (width as usize) * (height as usize)];
        let pan_x = f * 16;
        for y in 0..(height as usize) {
            for x in 0..(width as usize) {
                let rgb = if y < (height as usize * 3) / 4 {
                    let bar_idx = (((x + pan_x) % (width as usize)) * 7) / (width as usize);
                    bars_rgb[bar_idx.min(6)]
                } else {
                    let ramp = (x as f32) / (width as f32 - 1.0);
                    [ramp, ramp, ramp]
                };

                let phase = ((y & 1) << 1) | (x & 1);
                let sensor_linear = match phase {
                    0 => rgb[0] / wb_rb_levels[0],
                    1 | 2 => rgb[1],
                    _ => rgb[2] / wb_rb_levels[1],
                };
                let adu = 150.0 + sensor_linear.clamp(0.0, 1.0) * (4095.0 - 150.0);
                bayer[y * (width as usize) + x] = (adu.round() as u16).clamp(150, 4095);
            }
        }
        frames.push(bayer);
    }

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let bytes = build_synthetic_nev(width, height, &frames, wb_rb_levels, black_level, 800);
    std::fs::write(path, bytes)?;
    Ok(())
}
