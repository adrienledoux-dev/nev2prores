use std::fs;
use std::path::Path;

use crate::demuxer::NevDemuxedContainer;
use crate::encoder::ProResCodec;
use crate::metal_pipeline::ColorPipelineParams;

/// Generates a Final Cut Pro FCPXML v1.11 sidecar document referencing the transcoded
/// ProRes `.mov` asset with full Nikon camera & RAW processing metadata pre-populated.
pub fn generate_fcpxml_v1_11(
    mov_path: &Path,
    container: &NevDemuxedContainer,
    params: &ColorPipelineParams,
    codec: ProResCodec,
) -> String {
    let clip_name = mov_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Nikon_NRAW_Clip");

    let abs_mov = fs::canonicalize(mov_path).unwrap_or_else(|_| mov_path.to_path_buf());
    let file_url = format!("file://{}", xml_escape(&abs_mov.to_string_lossy()));

    let (out_w, out_h) = params
        .scale
        .output_dimensions(container.width, container.height);
    let (timescale, frame_dur, fps) = container.frame_rate();
    let total_dur = container
        .frames
        .iter()
        .map(|f| f.duration as u64)
        .sum::<u64>();

    // Compute starting rational time in seconds from `tmcd` track if present
    let start_ticks = container
        .timecode_track
        .as_ref()
        .map(|tc| (tc.frame_number as u64) * (frame_dur as u64))
        .unwrap_or(0);

    let has_audio = if container.audio_track.is_some() {
        1
    } else {
        0
    };
    let audio_channels = container
        .audio_track
        .as_ref()
        .map(|a| a.channels)
        .unwrap_or(2);
    let audio_rate = container
        .audio_track
        .as_ref()
        .map(|a| a.sample_rate)
        .unwrap_or(48000);

    let effective_kelvin = params
        .target_kelvin
        .unwrap_or(container.metadata.color_temperature_kelvin);
    let effective_tint = params.target_tint.unwrap_or(0.0);
    let shutter_str = container.metadata.shutter_display();
    let wb_gains = container.metadata.wb_rb_levels;

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.11">
    <resources>
        <format id="r1" name="FFVideoFormat{out_h}p{fps_int}" frameDuration="{frame_dur}/{timescale}s" width="{out_w}" height="{out_h}" colorSpace="9-1-9 (Rec. 2020)"/>
        <asset id="r2" name="{clip_name}" src="{file_url}" start="{start_ticks}/{timescale}s" duration="{total_dur}/{timescale}s" hasVideo="1" hasAudio="{has_audio}" format="r1" audioSources="1" audioChannels="{audio_channels}" audioRate="{audio_rate}" colorSpaceOverride="Rec. 2020 N-Log">
            <media-rep kind="original-media" src="{file_url}"/>
            <metadata>
                <md key="com.apple.proapps.mio.cameraName" value="{camera_model}"/>
                <md key="com.apple.proapps.mio.cameraISO" value="{iso}"/>
                <md key="com.apple.proapps.mio.shutterSpeed" value="{shutter_str}"/>
                <md key="com.apple.proapps.studio.rawToLogConversion" value="1"/>
                <md key="com.nikon.nraw.lensModel" value="{lens}"/>
                <md key="com.nikon.nraw.kelvin" value="{effective_kelvin}"/>
                <md key="com.nikon.nraw.tint" value="{effective_tint:.2}"/>
                <md key="com.nikon.nraw.evOffset" value="{ev:+.2}"/>
                <md key="com.nikon.nraw.asShotWBGains" value="{r_gain:.4},{b_gain:.4},{g1_gain:.4},{g2_gain:.4}"/>
                <md key="com.nikon.nraw.blackLevelsRGGB" value="{bl0},{bl1},{bl2},{bl3}"/>
                <md key="com.nikon.nraw.transcodeCodec" value="{codec_name}"/>
                <md key="com.nikon.nraw.transferFunction" value="Nikon N-Log"/>
                <md key="com.nikon.nraw.colorPrimaries" value="ITU-R BT.2020"/>
            </metadata>
        </asset>
    </resources>
    <library>
        <event name="nev2prores Transcodes">
            <project name="{clip_name}">
                <sequence format="r1" duration="{total_dur}/{timescale}s" tcStart="{start_ticks}/{timescale}s" tcFormat="NDF" audioLayout="stereo" audioRate="48k">
                    <spine>
                        <asset-clip name="{clip_name}" ref="r2" offset="{start_ticks}/{timescale}s" start="{start_ticks}/{timescale}s" duration="{total_dur}/{timescale}s" format="r1" tcFormat="NDF"/>
                    </spine>
                </sequence>
            </project>
        </event>
    </library>
</fcpxml>
"#,
        out_w = out_w,
        out_h = out_h,
        fps_int = fps.round() as u32,
        frame_dur = frame_dur,
        timescale = timescale,
        clip_name = xml_escape(clip_name),
        file_url = file_url,
        start_ticks = start_ticks,
        total_dur = total_dur,
        has_audio = has_audio,
        audio_channels = audio_channels,
        audio_rate = audio_rate,
        camera_model = xml_escape(&container.metadata.camera_model),
        iso = container.metadata.iso_speed,
        shutter_str = xml_escape(&shutter_str),
        lens = xml_escape(&container.metadata.lens_model),
        effective_kelvin = effective_kelvin,
        effective_tint = effective_tint,
        ev = params.ev_stops,
        r_gain = wb_gains[0],
        b_gain = wb_gains[1],
        g1_gain = wb_gains[2],
        g2_gain = wb_gains[3],
        bl0 = container.metadata.black_level[0],
        bl1 = container.metadata.black_level[1],
        bl2 = container.metadata.black_level[2],
        bl3 = container.metadata.black_level[3],
        codec_name = xml_escape(codec.display_name()),
    )
}

/// Writes the FCPXML v1.11 document to disk alongside the `.mov` file.
pub fn write_fcpxml_sidecar(
    fcpxml_path: &Path,
    mov_path: &Path,
    container: &NevDemuxedContainer,
    params: &ColorPipelineParams,
    codec: ProResCodec,
) -> std::io::Result<()> {
    let xml = generate_fcpxml_v1_11(mov_path, container, params, codec);
    fs::write(fcpxml_path, xml)
}

fn xml_escape(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c => out.push(c),
        }
    }
    out
}
