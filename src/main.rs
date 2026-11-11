use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};

use nev2prores::demuxer::demux_nev_file;
use nev2prores::encoder::ProResCodec;
use nev2prores::gui::run_native_gui;
use nev2prores::metal_pipeline::{MetalColorPipeline, ScaleMode};
use nev2prores::{
    generate_sample_nev_file_with_bit_depth, print_inspection_report, transcode_nev_file,
    TranscodeOptions,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CliCodec {
    #[value(name = "prores4444xq", alias = "ap4x")]
    Prores4444xq,
    #[value(name = "prores4444", alias = "ap4h")]
    Prores4444,
    #[value(name = "prores422hq", alias = "apch")]
    Prores422hq,
    #[value(name = "raw16", alias = "b64a")]
    Raw16,
}

impl From<CliCodec> for ProResCodec {
    fn from(c: CliCodec) -> Self {
        match c {
            CliCodec::Prores4444xq => ProResCodec::ProRes4444XQ,
            CliCodec::Prores4444 => ProResCodec::ProRes4444,
            CliCodec::Prores422hq => ProResCodec::ProRes422HQ,
            CliCodec::Raw16 => ProResCodec::Uncompressed16Bit,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum CliScale {
    #[value(name = "full")]
    Full,
    #[value(name = "half")]
    Half,
}

impl From<CliScale> for ScaleMode {
    fn from(s: CliScale) -> Self {
        match s {
            CliScale::Full => ScaleMode::Full,
            CliScale::Half => ScaleMode::Half,
        }
    }
}

/// Native macOS Nikon N-RAW (`.nev`) to 12-Bit Apple ProRes 4444 / N-Log / Rec. 2020 Transcoder
#[derive(Debug, Parser)]
#[command(
    name = "nev2prores",
    version,
    about = "Native Apple Silicon (Metal + VideoToolbox) Nikon Z8/Z9 N-RAW (.nev) to 12-Bit ProRes 4444 / N-Log Transcoder"
)]
pub struct CliArgs {
    /// Input Nikon `.nev` file or directory containing `.nev` files
    #[arg(long = "input", short = 'i', value_name = "FILE_OR_DIR")]
    pub input: Option<PathBuf>,

    /// Output directory for transcoded `.mov` and `.fcpxml` files
    #[arg(long = "output", short = 'o', value_name = "DIR")]
    pub output: Option<PathBuf>,

    /// Target mastering codec (`prores4444xq` = `ap4x`, `prores4444` = `ap4h`, `prores422hq` = `apch`, `raw16` = lossless 16-bit `b64a`)
    #[arg(
        long = "codec",
        value_enum,
        default_value = "prores4444",
        value_name = "prores4444xq|prores4444|prores422hq|raw16"
    )]
    pub codec: CliCodec,

    /// Optional target White Balance color temperature in Kelvin (e.g., 5600, 3200)
    #[arg(long = "kelvin", value_name = "OPTIONAL_TARGET_KELVIN")]
    pub kelvin: Option<u32>,

    /// Optional White Balance green/magenta tint shift (e.g., -10.0 to +10.0)
    #[arg(
        long = "tint",
        allow_hyphen_values = true,
        value_name = "OPTIONAL_TINT_SHIFT"
    )]
    pub tint: Option<f32>,

    /// Optional scene-linear exposure compensation in EV stops (e.g., +1.0, -0.5)
    #[arg(
        long = "ev",
        allow_hyphen_values = true,
        value_name = "OPTIONAL_EXPOSURE_STOPS_FLOAT"
    )]
    pub ev: Option<f32>,

    /// Output spatial resolution scale (`full` = 1:1 sensor resolution, `half` = 2x2 superpixel downsample)
    #[arg(
        long = "scale",
        value_enum,
        default_value = "full",
        value_name = "full|half"
    )]
    pub scale: CliScale,

    /// Emit a Final Cut Pro FCPXML v1.11 sidecar (`.fcpxml`) with camera & RAW metadata
    #[arg(long = "fcpxml")]
    pub fcpxml: bool,

    /// Inspect and print `.nev` container tracks, Nikon `NCDT` MakerNote tags, and color matrix without transcoding
    #[arg(long = "inspect-only")]
    pub inspect_only: bool,

    /// Generate a synthetic 1920x1080 RGGB `.nev` test file at `--input` before running
    #[arg(long = "generate-sample")]
    pub generate_sample: bool,

    /// Sensor bit depth (`12` or `14`) when using `--generate-sample`
    #[arg(long = "sample-bit-depth", default_value_t = 12)]
    pub sample_bit_depth: u16,

    /// Launch the native macOS Studio GUI (`egui` / Metal live RAW preview, Kelvin/Tint/EV sliders, and batch queue)
    #[arg(long = "gui")]
    pub gui: bool,
}

fn collect_nev_files(input: &Path) -> Result<Vec<PathBuf>> {
    if input.is_file() {
        return Ok(vec![input.to_path_buf()]);
    }
    if input.is_dir() {
        let mut files = Vec::new();
        for entry in std::fs::read_dir(input)
            .with_context(|| format!("Failed to read input directory '{}'", input.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if ext.eq_ignore_ascii_case("nev") {
                        files.push(path);
                    }
                }
            }
        }
        files.sort();
        if files.is_empty() {
            bail!("No .nev files found in directory '{}'", input.display());
        }
        return Ok(files);
    }
    bail!("Input path '{}' does not exist", input.display());
}

fn main() -> Result<()> {
    let args = CliArgs::parse();

    // Launch native macOS GUI if `--gui` is passed or if invoked with no `--input`
    if args.gui || args.input.is_none() {
        if let Some(ref input_path) = args.input {
            if args.generate_sample {
                generate_sample_nev_file_with_bit_depth(input_path, args.sample_bit_depth)?;
            }
        }
        return run_native_gui(args.input, args.output);
    }

    let input_path = args.input.as_ref().unwrap();
    if args.generate_sample {
        generate_sample_nev_file_with_bit_depth(input_path, args.sample_bit_depth)?;
        println!(
            "[nev2prores] Generated synthetic 1920x1080 {}-bit N-RAW sample (24 frames) at '{}'",
            if args.sample_bit_depth >= 14 { 14 } else { 12 },
            input_path.display()
        );
    }

    let nev_files = collect_nev_files(input_path)?;

    let opts = TranscodeOptions {
        codec: args.codec.into(),
        kelvin: args.kelvin,
        tint: args.tint,
        ev: args.ev,
        scale: args.scale.into(),
        emit_fcpxml: args.fcpxml,
        ring_buffer_capacity: 4,
    };

    if args.inspect_only {
        for nev_path in &nev_files {
            let container = demux_nev_file(nev_path)
                .with_context(|| format!("Failed to demux '{}'", nev_path.display()))?;
            print_inspection_report(nev_path, &container, &opts);
        }
        return Ok(());
    }

    let output_dir = match args.output {
        Some(ref dir) => dir.clone(),
        None => input_path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from(".")),
    };

    let mut gpu =
        MetalColorPipeline::new().context("Failed to initialize Apple Metal compute pipeline")?;

    println!(
        "[nev2prores] GPU Pipeline Initialized on: {} (Unified Memory StorageModeShared)",
        gpu.device_name()
    );
    println!(
        "[nev2prores] Target Codec: {} | Color Space: Nikon N-Log / ITU-R BT.2020",
        opts.codec.display_name()
    );

    for nev_path in &nev_files {
        println!("[nev2prores] Transcoding '{}' ...", nev_path.display());
        let report = transcode_nev_file(nev_path, &output_dir, &opts, &mut gpu)?;
        let fps = if report.elapsed_secs > 0.0 {
            report.frames_transcoded as f64 / report.elapsed_secs
        } else {
            0.0
        };
        println!(
            "[nev2prores] Completed '{}' -> '{}' ({} frames @ {}x{} in {:.2}s, {:.1} fps)",
            report.input_path.display(),
            report.output_mov_path.display(),
            report.frames_transcoded,
            report.output_width,
            report.output_height,
            report.elapsed_secs,
            fps
        );
        if let Some(ref fcpxml) = report.output_fcpxml_path {
            println!(
                "[nev2prores] Wrote FCPXML v1.11 sidecar: '{}'",
                fcpxml.display()
            );
        }
    }

    Ok(())
}
