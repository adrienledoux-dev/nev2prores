use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use crossbeam_channel::{unbounded, Receiver};
use eframe::egui::{
    self, Color32, ColorImage, Frame, Margin, RichText, Rounding, Stroke, TextureHandle,
    TextureOptions,
};

use crate::decoder::{NativeTicoRawDecoder, TicoRawDecoder};
use crate::demuxer::{demux_nev_file, NevDemuxedContainer};
use crate::encoder::ProResCodec;
use crate::metal_pipeline::{ColorPipelineParams, MetalColorPipeline, ScaleMode};
use crate::{
    generate_sample_nev_file, transcode_nev_file_with_progress, TranscodeOptions, TranscodeReport,
};

/// Embedded 256x256 PNG application icon used for the macOS Dock icon and in-app branding.
const APP_ICON_PNG: &[u8] = include_bytes!("../assets/logo_256.png");

#[derive(Debug, Clone)]
struct QueuedClip {
    path: PathBuf,
    container: NevDemuxedContainer,
}

#[derive(Debug, Clone)]
enum WorkerEvent {
    ClipStarted {
        clip_index: usize,
        total_clips: usize,
        clip_name: String,
    },
    FrameProgress {
        clip_index: usize,
        total_clips: usize,
        completed_frames: usize,
        total_frames: usize,
    },
    ClipFinished(TranscodeReport),
    BatchFinished {
        reports: Vec<TranscodeReport>,
    },
    Error(String),
}

/// Launches the native macOS `nev2prores` Studio GUI window with custom Dock icon and dark studio theme.
pub fn run_native_gui(
    initial_input: Option<PathBuf>,
    initial_output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let mut viewport = egui::ViewportBuilder::default()
        .with_title(
            "nev2prores Studio — Native macOS Nikon N-RAW (.nev) to 12-Bit ProRes 4444 / N-Log",
        )
        .with_inner_size([1400.0, 880.0])
        .with_min_inner_size([1080.0, 700.0])
        .with_drag_and_drop(true);

    if let Ok(icon_data) = eframe::icon_data::from_png_bytes(APP_ICON_PNG) {
        viewport = viewport.with_icon(Arc::new(icon_data));
    }

    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "nev2prores",
        native_options,
        Box::new(move |cc| {
            apply_studio_theme(&cc.egui_ctx, true);
            Ok(Box::new(Nev2ProResGuiApp::new(
                &cc.egui_ctx,
                initial_input,
                initial_output,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("Failed to launch eframe native GUI: {}", e))
}

/// Applies the custom Cinema Pro Dark Mode (or Light Mode) palette to `egui::Context`,
/// overriding macOS system theme events so the grading environment stays consistent.
fn apply_studio_theme(ctx: &egui::Context, dark_mode: bool) {
    if dark_mode {
        ctx.set_theme(egui::Theme::Dark);
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = Color32::from_rgb(20, 22, 27);
        visuals.window_fill = Color32::from_rgb(24, 27, 34);
        visuals.extreme_bg_color = Color32::from_rgb(12, 13, 16);
        visuals.faint_bg_color = Color32::from_rgb(28, 31, 39);
        visuals.code_bg_color = Color32::from_rgb(16, 18, 23);
        visuals.selection.bg_fill = Color32::from_rgb(215, 165, 0);
        visuals.selection.stroke = Stroke::new(1.0_f32, Color32::from_rgb(255, 220, 60));
        visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(26, 29, 36);
        visuals.widgets.noninteractive.bg_stroke =
            Stroke::new(1.0_f32, Color32::from_rgb(44, 49, 60));
        visuals.widgets.noninteractive.fg_stroke =
            Stroke::new(1.0_f32, Color32::from_rgb(215, 220, 230));
        visuals.widgets.inactive.bg_fill = Color32::from_rgb(34, 38, 48);
        visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(56, 62, 76));
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(230, 234, 242));
        visuals.widgets.hovered.bg_fill = Color32::from_rgb(48, 54, 68);
        visuals.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(245, 196, 0));
        visuals.widgets.active.bg_fill = Color32::from_rgb(62, 68, 86);
        visuals.widgets.active.bg_stroke = Stroke::new(1.5_f32, Color32::from_rgb(255, 210, 0));
        ctx.set_visuals(visuals);
    } else {
        ctx.set_theme(egui::Theme::Light);
        ctx.set_visuals(egui::Visuals::light());
    }
}

/// Renders a rich structured tooltip card displaying a technical summary along with Pros & Cons.
fn show_pros_cons_tooltip(
    ui: &mut egui::Ui,
    title: &str,
    summary: &str,
    pros: &[&str],
    cons: &[&str],
) {
    ui.set_max_width(400.0);
    ui.label(
        RichText::new(title)
            .strong()
            .color(Color32::from_rgb(255, 212, 0)),
    );
    ui.add_space(3.0);
    ui.label(summary);

    if !pros.is_empty() {
        ui.add_space(6.0);
        ui.label(
            RichText::new("✓ Pros:")
                .strong()
                .color(Color32::from_rgb(95, 225, 145)),
        );
        for &p in pros {
            ui.label(format!("  • {}", p));
        }
    }

    if !cons.is_empty() {
        ui.add_space(5.0);
        ui.label(
            RichText::new("✕ Cons / Trade-offs:")
                .strong()
                .color(Color32::from_rgb(255, 155, 100)),
        );
        for &c in cons {
            ui.label(format!("  • {}", c));
        }
    }
}

/// Renders a subtle `[?]` badge that displays a structured Pros & Cons tooltip card on hover.
fn info_badge_pros_cons(
    ui: &mut egui::Ui,
    title: &str,
    summary: &str,
    pros: &[&str],
    cons: &[&str],
) {
    ui.add(
        egui::Label::new(
            RichText::new("[?]")
                .small()
                .monospace()
                .color(Color32::from_rgb(140, 155, 185)),
        )
        .sense(egui::Sense::hover()),
    )
    .on_hover_ui(|ui| {
        show_pros_cons_tooltip(ui, title, summary, pros, cons);
    });
}

/// Styled container frame for sidebar cards.
fn studio_card_frame(dark_mode: bool) -> Frame {
    let fill = if dark_mode {
        Color32::from_rgb(27, 30, 38)
    } else {
        Color32::from_rgb(244, 245, 248)
    };
    let stroke = if dark_mode {
        Color32::from_rgb(46, 52, 66)
    } else {
        Color32::from_rgb(210, 214, 222)
    };
    Frame::none()
        .fill(fill)
        .stroke(Stroke::new(1.0_f32, stroke))
        .rounding(Rounding::same(6.0))
        .inner_margin(Margin::same(10.0))
}

pub struct Nev2ProResGuiApp {
    gpu: Option<MetalColorPipeline>,
    gpu_name: String,
    dark_mode: bool,
    logo_texture: Option<TextureHandle>,
    clips: Vec<QueuedClip>,
    selected_clip_idx: Option<usize>,
    output_dir: PathBuf,

    // Grading & Transcode Controls
    codec: ProResCodec,
    scale: ScaleMode,
    emit_fcpxml: bool,
    override_wb: bool,
    kelvin: u32,
    tint: f32,
    ev_stops: f32,
    display_rec709_lut: bool,

    // Live GPU Frame Preview State
    preview_frame_idx: usize,
    cached_bayer_key: Option<(PathBuf, usize)>,
    cached_bayer_rggb: Vec<u16>,
    cached_nlog_rgb: Vec<[f32; 3]>,
    preview_dirty: bool,
    preview_texture: Option<TextureHandle>,
    last_gpu_ms: f64,
    preview_resolution: (u32, u32),

    // Background Transcode Worker State
    worker_rx: Option<Receiver<WorkerEvent>>,
    is_transcoding: bool,
    progress_text: String,
    progress_fraction: f32,
    completed_reports: Vec<TranscodeReport>,
    status_banner: String,
}

impl Nev2ProResGuiApp {
    pub fn new(
        ctx: &egui::Context,
        initial_input: Option<PathBuf>,
        initial_output: Option<PathBuf>,
    ) -> Self {
        let (gpu, gpu_name) = match MetalColorPipeline::new() {
            Ok(p) => {
                let name = p.device_name();
                (Some(p), name)
            }
            Err(e) => (None, format!("Unavailable ({})", e)),
        };

        let logo_texture = eframe::icon_data::from_png_bytes(APP_ICON_PNG)
            .ok()
            .map(|icon| {
                let image = ColorImage::from_rgba_unmultiplied(
                    [icon.width as usize, icon.height as usize],
                    &icon.rgba,
                );
                ctx.load_texture("app_logo_badge", image, TextureOptions::LINEAR)
            });

        let default_out = initial_output.unwrap_or_else(|| {
            if let Ok(cwd) = std::env::current_dir() {
                if cwd != Path::new("/") {
                    return cwd.join("prores_out");
                }
            }
            if let Ok(home) = std::env::var("HOME") {
                return PathBuf::from(home).join("Movies").join("nev2prores");
            }
            std::env::temp_dir().join("nev2prores_out")
        });

        let mut app = Self {
            gpu,
            gpu_name,
            dark_mode: true,
            logo_texture,
            clips: Vec::new(),
            selected_clip_idx: None,
            output_dir: default_out,
            codec: ProResCodec::ProRes4444,
            scale: ScaleMode::Full,
            emit_fcpxml: true,
            override_wb: false,
            kelvin: 5600,
            tint: 0.0,
            ev_stops: 0.0,
            display_rec709_lut: false,
            preview_frame_idx: 0,
            cached_bayer_key: None,
            cached_bayer_rggb: Vec::new(),
            cached_nlog_rgb: Vec::new(),
            preview_dirty: true,
            preview_texture: None,
            last_gpu_ms: 0.0,
            preview_resolution: (0, 0),
            worker_rx: None,
            is_transcoding: false,
            progress_text: String::new(),
            progress_fraction: 0.0,
            completed_reports: Vec::new(),
            status_banner:
                "Ready. Open a Nikon .nev file/folder or click '▶ Generate Sample .NEV Clip' to preview on the Metal GPU."
                    .to_string(),
        };

        if let Some(path) = initial_input {
            app.add_path(&path);
        }

        app
    }

    fn add_path(&mut self, path: &Path) {
        if path.is_file() {
            match demux_nev_file(path) {
                Ok(container) => {
                    let as_shot_k = container.metadata.color_temperature_kelvin;
                    if let Some(existing_idx) = self.clips.iter().position(|c| c.path == path) {
                        self.clips[existing_idx].container = container;
                        self.select_clip(existing_idx, as_shot_k);
                    } else {
                        self.clips.push(QueuedClip {
                            path: path.to_path_buf(),
                            container,
                        });
                        let new_idx = self.clips.len() - 1;
                        self.select_clip(new_idx, as_shot_k);
                    }
                    self.status_banner = format!("Loaded '{}'", path.display());
                }
                Err(e) => {
                    self.status_banner = format!("Failed to demux '{}': {}", path.display(), e);
                }
            }
        } else if path.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                let mut paths: Vec<PathBuf> = entries
                    .flatten()
                    .map(|e| e.path())
                    .filter(|p| {
                        p.extension()
                            .and_then(|ext| ext.to_str())
                            .map(|ext| ext.eq_ignore_ascii_case("nev"))
                            .unwrap_or(false)
                    })
                    .collect();
                paths.sort();
                for p in paths {
                    self.add_path(&p);
                }
            }
        }
    }

    fn select_clip(&mut self, idx: usize, as_shot_kelvin: u32) {
        self.selected_clip_idx = Some(idx);
        self.preview_frame_idx = 0;
        if !self.override_wb {
            self.kelvin = as_shot_kelvin;
            self.tint = 0.0;
        }
        self.preview_dirty = true;
    }

    fn refresh_gpu_preview(&mut self, ctx: &egui::Context) {
        let Some(clip_idx) = self.selected_clip_idx else {
            return;
        };
        let Some(clip) = self.clips.get(clip_idx) else {
            return;
        };
        if clip.container.frames.is_empty() {
            return;
        }
        let frame_idx = self
            .preview_frame_idx
            .min(clip.container.frames.len().saturating_sub(1));
        let key = (clip.path.clone(), frame_idx);

        // 1. Decode TicoRAW RGGB frame on CPU only when clip or frame_idx changes
        if self.cached_bayer_key.as_ref() != Some(&key) {
            let desc = &clip.container.frames[frame_idx];
            let Ok(file) = File::open(&clip.path) else {
                return;
            };
            let mut reader = BufReader::new(file);
            let Ok(bitstream) = clip.container.read_frame_bytes(&mut reader, desc) else {
                return;
            };
            let mut decoder = NativeTicoRawDecoder::new();
            decoder.default_black_level = clip.container.metadata.black_level[0];
            let Ok(bayer) = decoder.decode_frame(
                &bitstream,
                clip.container.width as usize,
                clip.container.height as usize,
            ) else {
                return;
            };
            self.cached_bayer_rggb = bayer;
            self.cached_bayer_key = Some(key);
        }

        let Some(ref mut gpu) = self.gpu else {
            return;
        };

        // 2. Run Apple Metal Compute Shader (using Half scale if width >= 1280 for >60fps interactive scrubbing)
        let preview_scale = if clip.container.width >= 1280 {
            ScaleMode::Half
        } else {
            ScaleMode::Full
        };

        let params = ColorPipelineParams::from_metadata(
            clip.container.width,
            clip.container.height,
            &clip.container.metadata,
            preview_scale,
            if self.override_wb {
                Some(self.kelvin)
            } else {
                None
            },
            if self.override_wb {
                Some(self.tint)
            } else {
                None
            },
            Some(self.ev_stops),
        );

        let t0 = Instant::now();
        let Ok(rgba_f16) = gpu.process_frame_to_rgba16f_vec(&self.cached_bayer_rggb, &params)
        else {
            return;
        };
        self.last_gpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let (out_w, out_h) =
            preview_scale.output_dimensions(clip.container.width, clip.container.height);
        self.preview_resolution = (out_w, out_h);

        // 3. Convert RGBA16F N-Log buffer into 8-bit sRGB `egui::ColorImage` for viewport display
        let num_pixels = (out_w as usize) * (out_h as usize);
        let mut pixels = Vec::with_capacity(num_pixels);
        self.cached_nlog_rgb.clear();
        self.cached_nlog_rgb.reserve(num_pixels);
        let apply_lut = self.display_rec709_lut;

        for chunk in rgba_f16.as_chunks::<4>().0 {
            let r_nlog = chunk[0].to_f32().clamp(0.0, 1.0);
            let g_nlog = chunk[1].to_f32().clamp(0.0, 1.0);
            let b_nlog = chunk[2].to_f32().clamp(0.0, 1.0);
            self.cached_nlog_rgb.push([r_nlog, g_nlog, b_nlog]);

            let (r, g, b) = if apply_lut {
                (
                    nlog_to_display_gamma(r_nlog),
                    nlog_to_display_gamma(g_nlog),
                    nlog_to_display_gamma(b_nlog),
                )
            } else {
                (r_nlog, g_nlog, b_nlog)
            };

            pixels.push(Color32::from_rgb(
                (r * 255.0).round() as u8,
                (g * 255.0).round() as u8,
                (b * 255.0).round() as u8,
            ));
        }

        let image = ColorImage {
            size: [out_w as usize, out_h as usize],
            pixels,
        };

        if let Some(ref mut tex) = self.preview_texture {
            tex.set(image, TextureOptions::LINEAR);
        } else {
            self.preview_texture =
                Some(ctx.load_texture("metal_nlog_preview", image, TextureOptions::LINEAR));
        }

        self.preview_dirty = false;
    }

    fn start_transcode_job(&mut self, only_selected: bool) {
        if self.is_transcoding || self.clips.is_empty() {
            return;
        }

        let target_paths: Vec<PathBuf> = if only_selected {
            if let Some(idx) = self.selected_clip_idx {
                self.clips
                    .get(idx)
                    .map(|c| vec![c.path.clone()])
                    .unwrap_or_default()
            } else {
                return;
            }
        } else {
            self.clips.iter().map(|c| c.path.clone()).collect()
        };

        if target_paths.is_empty() {
            return;
        }

        let opts = TranscodeOptions {
            codec: self.codec,
            kelvin: if self.override_wb {
                Some(self.kelvin)
            } else {
                None
            },
            tint: if self.override_wb {
                Some(self.tint)
            } else {
                None
            },
            ev: Some(self.ev_stops),
            scale: self.scale,
            emit_fcpxml: self.emit_fcpxml,
            ring_buffer_capacity: 4,
        };
        let out_dir = self.output_dir.clone();

        let (tx, rx) = unbounded::<WorkerEvent>();
        self.worker_rx = Some(rx);
        self.is_transcoding = true;
        self.progress_fraction = 0.0;
        self.progress_text = "Initializing Apple Metal & VideoToolbox session...".to_string();

        thread::spawn(move || {
            let mut worker_gpu = match MetalColorPipeline::new() {
                Ok(g) => g,
                Err(e) => {
                    let _ = tx.send(WorkerEvent::Error(e.to_string()));
                    return;
                }
            };

            let total_clips = target_paths.len();
            let mut reports = Vec::with_capacity(total_clips);

            for (clip_index, path) in target_paths.iter().enumerate() {
                let clip_name = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("clip.nev")
                    .to_string();
                let _ = tx.send(WorkerEvent::ClipStarted {
                    clip_index,
                    total_clips,
                    clip_name,
                });

                let tx_prog = tx.clone();
                let res = transcode_nev_file_with_progress(
                    path,
                    &out_dir,
                    &opts,
                    &mut worker_gpu,
                    move |done, total| {
                        let _ = tx_prog.send(WorkerEvent::FrameProgress {
                            clip_index,
                            total_clips,
                            completed_frames: done,
                            total_frames: total,
                        });
                    },
                );

                match res {
                    Ok(report) => {
                        let _ = tx.send(WorkerEvent::ClipFinished(report.clone()));
                        reports.push(report);
                    }
                    Err(e) => {
                        let _ = tx.send(WorkerEvent::Error(format!("{:#}", e)));
                        return;
                    }
                }
            }

            let _ = tx.send(WorkerEvent::BatchFinished { reports });
        });
    }

    fn poll_worker_events(&mut self) {
        let Some(ref rx) = self.worker_rx else {
            return;
        };

        while let Ok(ev) = rx.try_recv() {
            match ev {
                WorkerEvent::ClipStarted {
                    clip_index,
                    total_clips,
                    clip_name,
                } => {
                    self.progress_text = format!(
                        "Transcoding '{}' ({}/{})...",
                        clip_name,
                        clip_index + 1,
                        total_clips
                    );
                }
                WorkerEvent::FrameProgress {
                    clip_index,
                    total_clips,
                    completed_frames,
                    total_frames,
                } => {
                    let clip_frac = if total_frames > 0 {
                        completed_frames as f32 / total_frames as f32
                    } else {
                        0.0
                    };
                    self.progress_fraction =
                        ((clip_index as f32) + clip_frac) / (total_clips.max(1) as f32);
                    self.progress_text = format!(
                        "Clip {}/{}: Frame {} / {} ({:.0}%)",
                        clip_index + 1,
                        total_clips,
                        completed_frames,
                        total_frames,
                        self.progress_fraction * 100.0
                    );
                }
                WorkerEvent::ClipFinished(report) => {
                    self.status_banner = format!(
                        "Completed '{}' ({} frames in {:.2}s)",
                        report.output_mov_path.display(),
                        report.frames_transcoded,
                        report.elapsed_secs
                    );
                    self.completed_reports.push(report);
                }
                WorkerEvent::BatchFinished { reports } => {
                    self.is_transcoding = false;
                    self.progress_fraction = 1.0;
                    self.progress_text = format!("Finished {} clip(s)", reports.len());
                }
                WorkerEvent::Error(err) => {
                    self.is_transcoding = false;
                    self.status_banner = format!("Error: {}", err);
                    self.progress_text = "Transcode failed".to_string();
                }
            }
        }
    }
}

/// Converts normalized Nikon N-Log `[0.0, 1.0]` back to scene-linear and applies a Rec.709 monitoring tone curve.
fn nlog_to_display_gamma(v: f32) -> f32 {
    let y = v * 1023.0;
    let linear = if y < 452.0 {
        let t = ((y - 128.0) / 650.0).max(0.0);
        (t * t * t - 0.0075).max(0.0)
    } else {
        ((y - 619.0) / 150.0).exp()
    };
    let mapped = linear / (1.0 + linear * 0.35);
    mapped.powf(1.0 / 2.2).clamp(0.0, 1.0)
}

impl eframe::App for Nev2ProResGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Enforce our selected Studio Dark/Light theme on every frame so OS theme notifications never override it
        apply_studio_theme(ctx, self.dark_mode);

        self.poll_worker_events();
        if self.is_transcoding {
            ctx.request_repaint();
        }

        // Handle drag-and-drop files onto the window
        let dropped: Vec<PathBuf> = ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .filter_map(|f| f.path.clone())
                .collect()
        });
        for path in dropped {
            self.add_path(&path);
        }

        if self.preview_dirty {
            self.refresh_gpu_preview(ctx);
        }

        // ====================================================================
        // Top Toolbar
        // ====================================================================
        egui::TopBottomPanel::top("top_toolbar").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if let Some(ref logo) = self.logo_texture {
                    ui.add(
                        egui::Image::new(logo)
                            .fit_to_exact_size(egui::vec2(26.0, 26.0))
                            .rounding(Rounding::same(5.0)),
                    );
                }
                ui.heading(
                    RichText::new("nev2prores")
                        .strong()
                        .color(Color32::from_rgb(255, 210, 0)),
                )
                .on_hover_ui(|ui| {
                    show_pros_cons_tooltip(
                        ui,
                        "nev2prores Studio (Native Rust + Metal + VideoToolbox)",
                        "Direct Apple Silicon hardware pipeline from Nikon N-RAW (.nev) to 12-bit ProRes 4444 / N-Log.",
                        &[
                            "Zero-copy Unified Memory between CPU TicoRAW unpacker and Apple Metal GPU",
                            "Scene-linear RAW Kelvin, Tint & EV adjustments prior to N-Log curve compression",
                            "Hardware-accelerated ProRes encoding via Apple VideoToolbox media engines",
                        ],
                        &[
                            "Requires macOS on Apple Silicon (M-series) for Unified Memory & ProRes hardware engines",
                        ],
                    );
                });
                ui.label(RichText::new("STUDIO").small().weak());

                ui.separator();

                if ui
                    .button("📂 Open .NEV File(s)...")
                    .on_hover_ui(|ui| {
                        show_pros_cons_tooltip(
                            ui,
                            "Open Individual .NEV Clip(s)",
                            "Select one or more Nikon Z8 / Z9 .nev files via the macOS file dialog.",
                            &[
                                "Inspect & grade specific hero takes individually",
                                "Supports multi-selection and drag-and-drop onto the window",
                            ],
                            &[
                                "For full camera cards with dozens of clips, 'Open Folder...' is faster",
                            ],
                        );
                    })
                    .clicked()
                {
                    if let Some(files) = rfd::FileDialog::new()
                        .add_filter("Nikon N-RAW (.nev)", &["nev", "NEV", "mov", "MOV"])
                        .pick_files()
                    {
                        for f in files {
                            self.add_path(&f);
                        }
                    }
                }

                if ui
                    .button("📁 Open Folder...")
                    .on_hover_ui(|ui| {
                        show_pros_cons_tooltip(
                            ui,
                            "Batch Import Folder / Camera Card",
                            "Scans an entire directory for .nev files and adds them to the clip queue.",
                            &[
                                "Fastest workflow for offloading an entire CFexpress Type B card",
                                "Pairs with 'Batch Transcode All Clips' for automated dailies generation",
                            ],
                            &[
                                "Applies the same global grading overrides when batch transcoding all clips at once",
                            ],
                        );
                    })
                    .clicked()
                {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.add_path(&dir);
                    }
                }

                if ui
                    .button("▶ Generate Sample .NEV Clip")
                    .on_hover_ui(|ui| {
                        show_pros_cons_tooltip(
                            ui,
                            "Synthetic 1920x1080 12-Bit RGGB Nikon Z9 Test Clip",
                            "Generates a 24-frame .nev container with SMPTE color bars, an exposure ramp, 24-bit PCM audio, tmcd timecode, and NCDT MakerNote tags.",
                            &[
                                "Instant verification of the Metal GPU debayer & VideoToolbox encoder without camera media",
                                "Includes known reference levels (75% color bars + 0..100% linear ramp) for scope checking",
                            ],
                            &[
                                "Synthetic 1080p test pattern (~75 MB in /tmp) rather than an 8K optical camera recording",
                            ],
                        );
                    })
                    .clicked()
                {
                    let sample_path = std::env::temp_dir().join("Nikon_Z9_Sample_1080p.NEV");
                    match generate_sample_nev_file(&sample_path) {
                        Ok(()) => self.add_path(&sample_path),
                        Err(e) => {
                            self.status_banner = format!("Sample generation error: {}", e);
                        }
                    }
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let theme_label = if self.dark_mode {
                        "🌙 Dark Studio"
                    } else {
                        "☀ Light Mode"
                    };
                    if ui
                        .button(theme_label)
                        .on_hover_ui(|ui| {
                            show_pros_cons_tooltip(
                                ui,
                                "Dark Studio vs. Light UI Appearance",
                                "Switches the grading interface surround luminance between Dark Studio (#14161B) and Light Mode.",
                                &[
                                    "Dark Studio: Eliminates bright UI surround glare so your eyes judge shadow contrast and color balance accurately",
                                    "Light Mode: Higher legibility in bright outdoor / on-set daylight environments",
                                ],
                                &[
                                    "Light Mode surround brightness makes video shadows appear artificially darker than they are (simultaneous contrast)",
                                ],
                            );
                        })
                        .clicked()
                    {
                        self.dark_mode = !self.dark_mode;
                    }

                    ui.separator();

                    ui.label(
                        RichText::new(format!("● Metal GPU: {}", self.gpu_name))
                            .small()
                            .monospace()
                            .color(Color32::from_rgb(95, 225, 145)),
                    )
                    .on_hover_ui(|ui| {
                        show_pros_cons_tooltip(
                            ui,
                            "Apple Silicon Unified Memory GPU Pipeline (StorageModeShared)",
                            "Zero-copy buffer sharing between CPU TicoRAW decoder and Metal compute kernels.",
                            &[
                                "Eliminates PCIe host-to-device memory copies across 8K RGGB frames",
                                "Writes directly into page-aligned CoreVideo CVPixelBuffer memory",
                            ],
                            &[
                                "Shares system RAM bandwidth between CPU performance cores and GPU compute units",
                            ],
                        );
                    });
                });
            });
            ui.add_space(6.0);
        });

        // ====================================================================
        // Bottom Status & Progress Bar
        // ====================================================================
        egui::TopBottomPanel::bottom("bottom_status").show(ctx, |ui| {
            ui.add_space(5.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(&self.status_banner).small());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some(last) = self.completed_reports.last() {
                        if ui
                            .button("🎬 Open .MOV in QuickTime")
                            .on_hover_text(
                                "Launch QuickTime Player to inspect the transcoded 12-bit ProRes 4444 / 422 HQ .mov file.",
                            )
                            .clicked()
                        {
                            let _ = Command::new("open").arg(&last.output_mov_path).spawn();
                        }
                        if ui
                            .button("📂 Reveal in Finder")
                            .on_hover_text(
                                "Highlight the transcoded .mov and .fcpxml sidecar files in macOS Finder.",
                            )
                            .clicked()
                        {
                            let _ = Command::new("open")
                                .arg("-R")
                                .arg(&last.output_mov_path)
                                .spawn();
                        }
                    }
                });
            });
            if self.is_transcoding || self.progress_fraction > 0.0 {
                ui.add_space(2.0);
                ui.add(
                    egui::ProgressBar::new(self.progress_fraction)
                        .text(&self.progress_text)
                        .animate(self.is_transcoding),
                );
            }
            ui.add_space(5.0);
        });

        // ====================================================================
        // Left Sidebar: Clip Queue & Nikon NCDT MakerNote Inspector
        // ====================================================================
        egui::SidePanel::left("left_inspector")
            .resizable(true)
            .default_width(345.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                studio_card_frame(self.dark_mode).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("1. Clip Queue");
                        info_badge_pros_cons(
                            ui,
                            "Stage 1: Zero-Copy 64-Bit ISO-BMFF Demuxer (src/demuxer.rs)",
                            "Parses moov -> trak -> mdia -> minf -> stbl (stsd, stts, stsc, stsz, co64) using Seek + Read.",
                            &[
                                "Opens multi-gigabyte >4GB .nev files in <1 ms without reading mdat into memory",
                                "Resolves exact 64-bit file offsets and frame durations for random-access scrubbing",
                            ],
                            &[
                                "Requires an intact 'moov' atom at the end of the .nev recording (uncorrupted camera file)",
                            ],
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if !self.clips.is_empty()
                                && ui
                                    .small_button("Clear")
                                    .on_hover_text("Remove all clips from the queue.")
                                    .clicked()
                            {
                                self.clips.clear();
                                self.selected_clip_idx = None;
                                self.preview_texture = None;
                            }
                        });
                    });
                    ui.separator();

                    let mut clicked_idx = None;
                    egui::ScrollArea::vertical()
                        .max_height(155.0)
                        .show(ui, |ui| {
                            if self.clips.is_empty() {
                                ui.weak(
                                    "No clips loaded.\nDrop .nev files here or click '▶ Generate Sample .NEV Clip'.",
                                );
                            }
                            for (idx, clip) in self.clips.iter().enumerate() {
                                let selected = self.selected_clip_idx == Some(idx);
                                let name = clip
                                    .path
                                    .file_name()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("clip.nev");
                                let (_, _, fps) = clip.container.frame_rate();
                                let label = format!(
                                    "{}  ({}x{}, {}f @ {:.2}fps)",
                                    name,
                                    clip.container.width,
                                    clip.container.height,
                                    clip.container.frames.len(),
                                    fps
                                );
                                if ui
                                    .selectable_label(selected, label)
                                    .on_hover_text(format!("Full Path: {}", clip.path.display()))
                                    .clicked()
                                {
                                    clicked_idx = Some((
                                        idx,
                                        clip.container.metadata.color_temperature_kelvin,
                                    ));
                                }
                            }
                        });

                    if let Some((idx, k)) = clicked_idx {
                        self.select_clip(idx, k);
                    }
                });

                ui.add_space(10.0);

                studio_card_frame(self.dark_mode).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("2. Nikon NCDT MakerNote");
                        info_badge_pros_cons(
                            ui,
                            "Stage 1.3: Nikon NCDT & MakerNote TIFF Parser (src/metadata.rs)",
                            "Extracts embedded 'Nikon\\0' MakerNote tags (0x000C WB_RBLevels, 0x003D BlackLevel, ISO, Shutter, Lens).",
                            &[
                                "Reads true sensor optical black levels and As-Shot R/B white balance multipliers",
                                "Supports both little-endian ('II') and big-endian ('MM') embedded TIFF headers",
                            ],
                            &[
                                "Proprietary encrypted Nikon lens tags fall back to standard Exif lens descriptors if absent",
                            ],
                        );
                    });
                    ui.separator();

                    if let Some(idx) = self.selected_clip_idx {
                        if let Some(clip) = self.clips.get(idx) {
                            let c = &clip.container;
                            let (_, _, fps) = c.frame_rate();
                            egui::Grid::new("metadata_grid")
                                .num_columns(2)
                                .spacing([12.0, 6.0])
                                .striped(true)
                                .show(ui, |ui| {
                                    ui.label("Camera Body:")
                                        .on_hover_text("TIFF Tag 0x0110 (Model) extracted from NCDT.");
                                    ui.strong(&c.metadata.camera_model);
                                    ui.end_row();

                                    ui.label("Lens:")
                                        .on_hover_text("Nikon MakerNote Tag 0x0084 / Exif Tag 0xA434 (LensModel).");
                                    ui.label(&c.metadata.lens_model);
                                    ui.end_row();

                                    ui.label("Sensor Format:")
                                        .on_hover_text("Dimensions and Bayer CFA bit depth from video track 'stsd' atom.");
                                    ui.label(format!(
                                        "{}x{} ({}-bit RGGB)",
                                        c.width, c.height, c.bit_depth
                                    ));
                                    ui.end_row();

                                    ui.label("Frame Rate:")
                                        .on_hover_text("Computed from 'mdhd' timescale and 'stts' sample duration.");
                                    ui.label(format!("{:.3} fps ({} frames)", fps, c.frames.len()));
                                    ui.end_row();

                                    ui.label("ISO / Shutter:")
                                        .on_hover_text("Nikon Tag 0x0002 (ISO) and Exif Tag 0x829A (ExposureTime).");
                                    ui.label(format!(
                                        "ISO {}  |  {}",
                                        c.metadata.iso_speed,
                                        c.metadata.shutter_display()
                                    ));
                                    ui.end_row();

                                    ui.label("Timecode (tmcd):")
                                        .on_hover_text(
                                            "Original SMPTE camera timecode track (tmcd) passed through into the output .mov and FCPXML.",
                                        );
                                    ui.monospace(c.metadata.timecode.as_deref().unwrap_or("None"));
                                    ui.end_row();

                                    ui.label("Audio Track:")
                                        .on_hover_text(
                                            "Uncompressed 24-bit Linear PCM audio track copied bit-for-bit into the output .mov container.",
                                        );
                                    if let Some(ref a) = c.audio_track {
                                        ui.label(format!(
                                            "{}-bit PCM, {}ch @ {} Hz",
                                            a.sample_bits, a.channels, a.sample_rate
                                        ));
                                    } else {
                                        ui.label("None");
                                    }
                                    ui.end_row();

                                    ui.label("WB_RBLevels (0x000C):")
                                        .on_hover_text(
                                            "Nikon MakerNote Tag 0x000C: 4 rationals [R_gain, B_gain, G1_gain, G2_gain]\n\
                                             representing the As-Shot sensor white balance multipliers in scene-linear light.",
                                        );
                                    ui.monospace(format!(
                                        "R:{:.3} B:{:.3} G:{:.2}",
                                        c.metadata.wb_rb_levels[0],
                                        c.metadata.wb_rb_levels[1],
                                        c.metadata.wb_rb_levels[2]
                                    ));
                                    ui.end_row();

                                    ui.label("BlackLevel (0x003D):")
                                        .on_hover_text(
                                            "Nikon MakerNote Tag 0x003D: 4 u16 RGGB optical black subtraction levels in ADU,\n\
                                             normalized against WhiteLevel (4095 for 12-bit, 16383 for 14-bit) in the Metal shader.",
                                        );
                                    ui.monospace(format!(
                                        "{:?} / {}",
                                        c.metadata.black_level, c.metadata.white_level
                                    ));
                                    ui.end_row();
                                });

                            ui.add_space(10.0);
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Bradford Adaptation Matrix M_WB").strong());
                                info_badge_pros_cons(
                                    ui,
                                    "3x3 Bradford Chromatic Adaptation Matrix (M_WB)",
                                    "M_WB = M_(XYZ->Rec2020) * M_B^(-1) * diag(M_B*W_src / M_B*W_dst) * M_B * M_(Camera->XYZ)",
                                    &[
                                        "Models human cone response (LMS) more accurately than simple RGB channel scaling",
                                        "Fuses Camera->XYZ, white point adaptation, and XYZ->Rec.2020 into a single 3x3 GPU matrix multiply",
                                    ],
                                    &[
                                        "Extreme Kelvin shifts (<2200K or >9500K) can push saturated primaries near the Rec.2020 gamut boundary",
                                    ],
                                );
                            });
                            let params = ColorPipelineParams::from_metadata(
                                c.width,
                                c.height,
                                &c.metadata,
                                self.scale,
                                if self.override_wb {
                                    Some(self.kelvin)
                                } else {
                                    None
                                },
                                if self.override_wb {
                                    Some(self.tint)
                                } else {
                                    None
                                },
                                Some(self.ev_stops),
                            );
                            let u = params.build_uniforms(c.width);
                            ui.monospace(format!(
                                "[ {:+.4}  {:+.4}  {:+.4} ]\n[ {:+.4}  {:+.4}  {:+.4} ]\n[ {:+.4}  {:+.4}  {:+.4} ]",
                                u.color_matrix_row0[0],
                                u.color_matrix_row0[1],
                                u.color_matrix_row0[2],
                                u.color_matrix_row1[0],
                                u.color_matrix_row1[1],
                                u.color_matrix_row1[2],
                                u.color_matrix_row2[0],
                                u.color_matrix_row2[1],
                                u.color_matrix_row2[2],
                            ));
                        }
                    } else {
                        ui.weak("Select a clip in the queue above to inspect its Nikon NCDT MakerNote tags.");
                    }
                });
            });

        // ====================================================================
        // Right Sidebar: Scene-Linear RAW Controls & ProRes Export Settings
        // ====================================================================
        egui::SidePanel::right("right_controls")
            .resizable(true)
            .default_width(355.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                studio_card_frame(self.dark_mode).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("3. Scene-Linear RAW Grading");
                        info_badge_pros_cons(
                            ui,
                            "Stage 3: Scene-Linear RAW Color Pipeline (shaders/pipeline.metal)",
                            "Evaluates WB gains, EV exposure stops, and Bradford chromatic adaptation in 32-bit float scene-linear light before Nikon N-Log encoding.",
                            &[
                                "Mathematically equivalent to changing ISO or Kelvin in-camera on the RAW sensor",
                                "Preserves full highlight roll-off and shadow linearity prior to log compression",
                            ],
                            &[
                                "Adjustments are baked into the exported ProRes N-Log master (though 12-bit ProRes 4444 retains wide grading latitude)",
                            ],
                        );
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .small_button("↺ Reset")
                                .on_hover_text("Reset White Balance and Exposure (EV) to As-Shot camera defaults.")
                                .clicked()
                            {
                                self.override_wb = false;
                                self.ev_stops = 0.0;
                                if let Some(idx) = self.selected_clip_idx {
                                    if let Some(c) = self.clips.get(idx) {
                                        self.kelvin = c.container.metadata.color_temperature_kelvin;
                                    }
                                }
                                self.tint = 0.0;
                                self.preview_dirty = true;
                            }
                        });
                    });
                    ui.separator();

                    if ui
                        .checkbox(&mut self.override_wb, "Custom White Balance Override (Bradford CAT)")
                        .on_hover_ui(|ui| {
                            show_pros_cons_tooltip(
                                ui,
                                "Custom White Balance (Bradford CAT) vs. As-Shot WB",
                                "Controls whether the shader uses only the camera's As-Shot WB_RBLevels or applies a 3x3 Bradford chromatic adaptation transform.",
                                &[
                                    "Unchecked (As-Shot): Preserves exact on-set camera white balance recorded in NCDT Tag 0x000C",
                                    "Checked (Custom): Allows lossless RAW re-balancing to any Kelvin (2000K–10000K) & Tint in scene-linear light",
                                ],
                                &[
                                    "Overriding WB changes the scene-linear channel ratios before N-Log encoding; verify skin tones in the viewport",
                                ],
                            );
                        })
                        .changed()
                    {
                        self.preview_dirty = true;
                    }

                    ui.add_enabled_ui(self.override_wb, |ui| {
                        ui.add_space(4.0);
                        ui.label("Color Temperature (Kelvin):")
                            .on_hover_ui(|ui| {
                                show_pros_cons_tooltip(
                                    ui,
                                    "Correlated Color Temperature (2000 K – 10000 K)",
                                    "Adapts the scene illuminant along the Planckian / CIE daylight locus.",
                                    &[
                                        "No channel clipping or quantization banding compared to grading WB after log encoding",
                                        "Updates the 3x3 Bradford matrix M_WB in real time on the GPU",
                                    ],
                                    &[
                                        "Very high Kelvin (>8000K) boosts Red gain, which can push bright warm highlights closer to clipping",
                                    ],
                                );
                            });
                        if ui
                            .add(
                                egui::Slider::new(&mut self.kelvin, 2000..=10000)
                                    .suffix(" K")
                                    .step_by(50.0),
                            )
                            .on_hover_ui(|ui| {
                                show_pros_cons_tooltip(
                                    ui,
                                    "Kelvin White Balance Slider",
                                    "Drag to warm or cool the image in scene-linear light.",
                                    &[
                                        "Lower Kelvin (e.g. 3200K) cools warm tungsten light",
                                        "Higher Kelvin (e.g. 6500K) warms cool shade/daylight",
                                    ],
                                    &[
                                        "Large shifts should be paired with Tint adjustment for mixed-spectrum LED/fluorescent sources",
                                    ],
                                );
                            })
                            .changed()
                        {
                            self.preview_dirty = true;
                        }

                        ui.horizontal(|ui| {
                            for &(label, k, title, summary, pro, con) in &[
                                (
                                    "3200K",
                                    3200u32,
                                    "3200 K — Tungsten / Incandescent Preset",
                                    "Standard indoor halogen & tungsten studio lighting balance.",
                                    "Neutralizes warm orange cast from 3200K tungsten lamps",
                                    "Makes daylight-lit scenes appear strongly blue/cold",
                                ),
                                (
                                    "4300K",
                                    4300,
                                    "4300 K — Mixed / Fluorescent Preset",
                                    "Intermediate balance for warm-white office/commercial lighting or mixed tungsten+daylight.",
                                    "Balanced compromise when shooting under mixed indoor/window light",
                                    "Often requires a slight Green/Magenta Tint tweak for non-blackbody fluorescent tubes",
                                ),
                                (
                                    "5600K",
                                    5600,
                                    "5600 K — Direct Daylight / HMI Preset",
                                    "Cinema standard direct sun and HMI lighting white point.",
                                    "Matches standard cinema daylight fixtures and sunny outdoor shooting",
                                    "Can look slightly cool under overcast skies or open shade",
                                ),
                                (
                                    "6500K",
                                    6500,
                                    "6500 K — Overcast / CIE D65 Preset",
                                    "Native white point of ITU-R BT.2020 / Rec. 709 (D65) and overcast daylight.",
                                    "Warms up cool outdoor shade and aligns with the D65 display reference",
                                    "Can make direct golden-hour sunlight appear overly warm/amber",
                                ),
                            ] {
                                if ui
                                    .small_button(label)
                                    .on_hover_ui(|ui| {
                                        show_pros_cons_tooltip(ui, title, summary, &[pro], &[con]);
                                    })
                                    .clicked()
                                {
                                    self.kelvin = k;
                                    self.preview_dirty = true;
                                }
                            }
                        });

                        ui.add_space(6.0);
                        ui.label("Green / Magenta Tint Shift:")
                            .on_hover_ui(|ui| {
                                show_pros_cons_tooltip(
                                    ui,
                                    "Green / Magenta Isothermal Tint Shift (-50 to +50)",
                                    "Shifts chromaticity perpendicular to the Planckian locus in CIE 1960 UCS (u, v) space.",
                                    &[
                                        "Corrects green spikes in LED / fluorescent fixtures (+Tint toward Magenta)",
                                        "Corrects magenta casts in skin tones (-Tint toward Green)",
                                    ],
                                    &[
                                        "Shifts > ±20 will introduce a noticeable green or magenta cast to neutral grays",
                                    ],
                                );
                            });
                        if ui
                            .add(
                                egui::Slider::new(&mut self.tint, -50.0..=50.0)
                                    .step_by(0.5),
                            )
                            .on_hover_ui(|ui| {
                                show_pros_cons_tooltip(
                                    ui,
                                    "Green / Magenta Tint Slider",
                                    "Fine-tunes green vs. magenta balance in scene-linear light.",
                                    &[
                                        "0.0 preserves pure Planckian blackbody locus adaptation",
                                        "Sub-unit precision (0.5 steps) for accurate skin-tone matching",
                                    ],
                                    &[
                                        "Should be kept near 0.0 under natural sunlight or high-CRI tungsten sources",
                                    ],
                                );
                            })
                            .changed()
                        {
                            self.preview_dirty = true;
                        }
                    });

                    ui.add_space(10.0);
                    ui.label(RichText::new("Scene-Linear Exposure Offset (ΔEV):").strong())
                        .on_hover_ui(|ui| {
                            show_pros_cons_tooltip(
                                ui,
                                "Scene-Linear Exposure Compensation (2^ΔEV)",
                                "Multiplies linear sensor RGB by 2^ΔEV prior to Nikon N-Log curve compression.",
                                &[
                                    "Exact photometric stop adjustment (+1.0 EV = 2.0x light, -1.0 EV = 0.5x light)",
                                    "Recovers shadow detail or places middle gray at target N-Log IRE (35%) before ProRes quantization",
                                ],
                                &[
                                    "Positive EV (+EV) reduces headroom above 100% reflection; negative EV (-EV) lowers the sensor ceiling",
                                ],
                            );
                        });
                    ui.horizontal(|ui| {
                        if ui
                            .add(
                                egui::Slider::new(&mut self.ev_stops, -4.0..=4.0)
                                    .suffix(" EV")
                                    .step_by(0.05),
                            )
                            .on_hover_ui(|ui| {
                                show_pros_cons_tooltip(
                                    ui,
                                    "Exposure Offset Slider (-4.00 EV to +4.00 EV)",
                                    "Scales linear RGB before the Nikon N-Log transfer function.",
                                    &[
                                        "Cleanly lifts underexposed N-RAW takes without post-log LUT posterization",
                                        "Real-time GPU preview updates in <2 ms",
                                    ],
                                    &[
                                        "Any sensor pixel already clipped at WhiteLevel (4095/16383) in the .nev cannot recover detail with -EV",
                                    ],
                                );
                            })
                            .changed()
                        {
                            self.preview_dirty = true;
                        }
                        if ui
                            .small_button("0.0 EV")
                            .on_hover_text("Reset exposure compensation to 0.0 EV (unity gain).")
                            .clicked()
                        {
                            self.ev_stops = 0.0;
                            self.preview_dirty = true;
                        }
                    });
                });

                ui.add_space(10.0);

                studio_card_frame(self.dark_mode).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.heading("4. ProRes & FCPXML Export");
                        info_badge_pros_cons(
                            ui,
                            "Stage 4: Hardware VideoToolbox ProRes & QuickTime Muxer",
                            "Encodes via Apple Silicon ProRes hardware engines and writes a 64-bit .mov with Rec.2020 'colr' ('nclc') tags.",
                            &[
                                "Dedicated Apple Silicon ProRes media engine offloads CPU & GPU during encoding",
                                "Passes through uncompressed 24-bit Linear PCM audio and original 'tmcd' camera timecode",
                            ],
                            &[
                                "All-intra ProRes masters require fast SSD storage for 8K high-bitrate output",
                            ],
                        );
                    });
                    ui.separator();

                    ui.label(RichText::new("Hardware VideoToolbox Codec:").strong());
                    ui.radio_value(
                        &mut self.codec,
                        ProResCodec::ProRes4444,
                        "Apple ProRes 4444 (ap4h, 12-bit 4:4:4)",
                    )
                    .on_hover_ui(|ui| {
                        show_pros_cons_tooltip(
                            ui,
                            "Apple ProRes 4444 ('ap4h' — 12-Bit 4:4:4)",
                            "Highest-fidelity Apple ProRes profile with unsubsampled 4:4:4 chroma and 12-bit sample precision.",
                            &[
                                "Full 12-bit precision matches Nikon Z8/Z9 12-bit N-RAW sensor depth without bit-depth loss",
                                "Unsubsampled 4:4:4 color is ideal for green-screen keying, VFX, and heavy N-Log color grading",
                            ],
                            &[
                                "Approximately ~50% larger file size and higher disk bandwidth than ProRes 422 HQ",
                            ],
                        );
                    });
                    ui.radio_value(
                        &mut self.codec,
                        ProResCodec::ProRes422HQ,
                        "Apple ProRes 422 HQ (apch, 10-bit 4:2:2)",
                    )
                    .on_hover_ui(|ui| {
                        show_pros_cons_tooltip(
                            ui,
                            "Apple ProRes 422 HQ ('apch' — 10-Bit 4:2:2)",
                            "Industry-standard broadcast and editorial mastering codec with 10-bit 4:2:2 chroma.",
                            &[
                                "Significantly smaller `.mov` file sizes for faster storage and timeline playback",
                                "Universal compatibility across all NLEs and broadcast delivery specs",
                            ],
                            &[
                                "10-bit quantization (vs. 12-bit in 4444) and horizontal 4:2:2 chroma subsampling",
                            ],
                        );
                    });

                    ui.add_space(8.0);
                    ui.label(RichText::new("Output Spatial Resolution:").strong());
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut self.scale, ScaleMode::Full, "Full (1:1 Native)")
                            .on_hover_ui(|ui| {
                                show_pros_cons_tooltip(
                                    ui,
                                    "Full Resolution (1:1 Native Sensor Dimensions)",
                                    "Reconstructs every Bayer pixel (e.g., 8256x4644 8K or 1920x1080) using 5x5 Malvar-He-Cutler demosaicing.",
                                    &[
                                        "Preserves 100% of native sensor spatial resolution and fine texture detail",
                                        "Allows reframing, stabilization, and 8K/4K mastering without upscaling",
                                    ],
                                    &[
                                        "4x larger pixel count and file size compared to Half-resolution proxy mode",
                                    ],
                                );
                            });
                        ui.radio_value(&mut self.scale, ScaleMode::Half, "Half (1/2 Proxy)")
                            .on_hover_ui(|ui| {
                                show_pros_cons_tooltip(
                                    ui,
                                    "Half Resolution (2x2 Superpixel Box-Filtered Downsample)",
                                    "Evaluates 2x2 Malvar-He-Cutler RGGB quads on the GPU to output half width & height (e.g., 8K -> 4K).",
                                    &[
                                        "~4x faster transcoding and ~75% smaller `.mov` files for lightweight offline/proxy editing",
                                        "2x2 superpixel averaging naturally reduces high-ISO sensor chroma noise",
                                    ],
                                    &[
                                        "Halves horizontal and vertical spatial resolution compared to the native sensor",
                                    ],
                                );
                            });
                    });

                    ui.add_space(8.0);
                    ui.checkbox(
                        &mut self.emit_fcpxml,
                        "Generate Final Cut Pro FCPXML v1.11 (.fcpxml)",
                    )
                    .on_hover_ui(|ui| {
                        show_pros_cons_tooltip(
                            ui,
                            "Final Cut Pro FCPXML v1.11 Sidecar (.fcpxml)",
                            "Writes a companion XML file alongside the .mov with camera & RAW metadata pre-populated.",
                            &[
                                "Automatically tags the clip as 'Rec. 2020 N-Log' in Final Cut Pro upon import",
                                "Populates Inspector metadata (ISO, Shutter Speed, Lens, Kelvin, Tint, ΔEV, Timecode)",
                            ],
                            &[
                                "Creates an extra `.fcpxml` file next to each `.mov` in the output directory (not needed if editing only in Resolve/Premiere)",
                            ],
                        );
                    });

                    ui.add_space(8.0);
                    ui.label(RichText::new("Output Folder:").strong());
                    ui.monospace(self.output_dir.to_string_lossy())
                        .on_hover_text("Destination directory where transcoded .mov and .fcpxml files will be written.");
                    if ui
                        .button("📁 Choose Output Folder...")
                        .on_hover_text("Select a destination folder using the macOS folder dialog.")
                        .clicked()
                    {
                        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                            self.output_dir = dir;
                        }
                    }

                    ui.add_space(12.0);
                    ui.separator();

                    let can_run = !self.is_transcoding && !self.clips.is_empty();
                    ui.add_enabled_ui(can_run, |ui| {
                        if ui
                            .add_sized(
                                [ui.available_width(), 38.0],
                                egui::Button::new(
                                    RichText::new("⚡ Transcode Selected Clip to ProRes")
                                        .strong()
                                        .color(Color32::from_rgb(255, 215, 0)),
                                ),
                            )
                            .on_hover_ui(|ui| {
                                show_pros_cons_tooltip(
                                    ui,
                                    "Transcode Selected Clip",
                                    "Executes the 4-stage pipeline on the currently active clip in the viewport.",
                                    &[
                                        "Fastest way to export a single hero shot after dialing in Kelvin, Tint & EV",
                                        "Runs asynchronously on background threads with live frame progress",
                                    ],
                                    &[
                                        "Only exports the currently selected clip (use 'Batch Transcode' below when multiple clips are queued)",
                                    ],
                                );
                            })
                            .clicked()
                        {
                            self.start_transcode_job(true);
                        }
                        if self.clips.len() > 1 {
                            ui.add_space(6.0);
                            if ui
                                .add_sized(
                                    [ui.available_width(), 32.0],
                                    egui::Button::new(format!(
                                        "⚡ Batch Transcode All {} Clips",
                                        self.clips.len()
                                    )),
                                )
                                .on_hover_ui(|ui| {
                                    show_pros_cons_tooltip(
                                        ui,
                                        "Batch Transcode All Queued Clips",
                                        "Transcodes every clip in the queue sequentially to the output folder.",
                                        &[
                                            "Unattended processing for an entire shoot or camera card",
                                            "Preserves each clip's individual As-Shot WB unless Custom WB Override is checked",
                                        ],
                                        &[
                                            "If Custom WB Override or ΔEV is active, that override is applied uniformly to all clips in the batch",
                                        ],
                                    );
                                })
                                .clicked()
                            {
                                self.start_transcode_job(false);
                            }
                        }
                    });
                });
            });

        // ====================================================================
        // Center Panel: Live Metal GPU Debayer & N-Log Viewport
        // ====================================================================
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Apple Metal GPU Viewport");
                info_badge_pros_cons(
                    ui,
                    "Real-Time Apple Metal Compute Viewport",
                    "Executes shaders/pipeline.metal on the GPU and displays the resulting frame.",
                    &[
                        "Sub-2ms GPU compute latency enables 60+ fps interactive Kelvin, Tint, and EV grading",
                        "Hover anywhere over the video frame to inspect exact 10-bit N-Log [0..1023] code values",
                    ],
                    &[
                        "Uses Half-scale preview for >1280px clips to guarantee instantaneous slider response",
                    ],
                );
                ui.separator();
                if ui
                    .selectable_label(!self.display_rec709_lut, "Nikon N-Log (Encoded Signal)")
                    .on_hover_ui(|ui| {
                        show_pros_cons_tooltip(
                            ui,
                            "Nikon N-Log (Flat Logarithmic Encoded Signal)",
                            "Displays the exact un-tone-mapped Nikon N-Log / Rec. 2020 signal written into the ProRes .mov container.",
                            &[
                                "Shows true log signal distribution (Black = 255 / 25% IRE, 18% Gray = 362 / 35% IRE)",
                                "Makes it easy to spot sensor clipping or crushed blacks before encoding",
                            ],
                            &[
                                "Appears flat, low-contrast, and desaturated on a standard monitor because no display LUT is applied",
                            ],
                        );
                    })
                    .clicked()
                {
                    self.display_rec709_lut = false;
                    self.preview_dirty = true;
                }
                if ui
                    .selectable_label(self.display_rec709_lut, "Rec. 709 Display LUT (Preview)")
                    .on_hover_ui(|ui| {
                        show_pros_cons_tooltip(
                            ui,
                            "Rec. 709 Display LUT (Non-Destructive Monitoring Preview)",
                            "Applies a viewing transform (N-Log inverse -> filmic highlight shoulder -> Gamma 2.2) in the viewport ONLY.",
                            &[
                                "Natural contrast and color saturation for judging skin tones, Kelvin, and Tint by eye",
                                "100% non-destructive: the exported ProRes .mov file is still encoded in pure Nikon N-Log / Rec. 2020",
                            ],
                            &[
                                "Compresses extreme highlight roll-off visually in the preview compared to the raw N-Log signal view",
                            ],
                        );
                    })
                    .clicked()
                {
                    self.display_rec709_lut = true;
                    self.preview_dirty = true;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.preview_resolution.0 > 0 {
                        ui.label(
                            RichText::new(format!(
                                "5x5 Malvar-He-Cutler + N-Log ({}x{}): {:.2} ms",
                                self.preview_resolution.0,
                                self.preview_resolution.1,
                                self.last_gpu_ms
                            ))
                            .small()
                            .monospace(),
                        )
                        .on_hover_text("Execution time of the last Metal compute shader dispatch on the GPU.");
                    }
                });
            });
            ui.separator();

            if let Some(ref tex) = self.preview_texture {
                let avail = ui.available_size() - egui::vec2(0.0, 42.0);
                let tex_size = tex.size_vec2();
                let aspect = tex_size.x / tex_size.y.max(1.0);
                let mut draw_w = avail.x;
                let mut draw_h = draw_w / aspect;
                if draw_h > avail.y {
                    draw_h = avail.y;
                    draw_w = draw_h * aspect;
                }

                Frame::none()
                    .fill(Color32::from_rgb(10, 11, 14))
                    .stroke(Stroke::new(1.0_f32, Color32::from_rgb(40, 44, 54)))
                    .rounding(Rounding::same(6.0))
                    .inner_margin(Margin::same(6.0))
                    .show(ui, |ui| {
                        ui.vertical_centered(|ui| {
                            let img_resp = ui.add(
                                egui::Image::new(tex)
                                    .fit_to_exact_size(egui::vec2(draw_w, draw_h))
                                    .sense(egui::Sense::hover()),
                            );

                            // Live Pixel Color / 10-bit N-Log Code Value Inspector on Hover
                            if let Some(hover_pos) = img_resp.hover_pos() {
                                let rect = img_resp.rect;
                                let u = ((hover_pos.x - rect.left()) / rect.width().max(1.0))
                                    .clamp(0.0, 0.9999);
                                let v = ((hover_pos.y - rect.top()) / rect.height().max(1.0))
                                    .clamp(0.0, 0.9999);
                                let px = (u * (self.preview_resolution.0 as f32)) as usize;
                                let py = (v * (self.preview_resolution.1 as f32)) as usize;
                                let idx = py * (self.preview_resolution.0 as usize) + px;
                                if let Some(&[r, g, b]) = self.cached_nlog_rgb.get(idx) {
                                    let r10 = (r * 1023.0).round() as u32;
                                    let g10 = (g * 1023.0).round() as u32;
                                    let b10 = (b * 1023.0).round() as u32;
                                    img_resp.on_hover_text(format!(
                                        "Pixel ({}, {})\n\
                                         Nikon N-Log Float : R={:.4}  G={:.4}  B={:.4}\n\
                                         10-Bit Code Value : R={:4}    G={:4}    B={:4}  (out of 1023)",
                                        px, py, r, g, b, r10, g10, b10
                                    ));
                                }
                            }
                        });
                    });

                // Frame Scrubber at bottom of viewport
                if let Some(idx) = self.selected_clip_idx {
                    if let Some(clip) = self.clips.get(idx) {
                        let num_frames = clip.container.frames.len();
                        if num_frames > 1 {
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.label("Timeline Scrubber:")
                                    .on_hover_text("Scrub to any frame in the .nev container; unpacks TicoRAW on CPU and debayers on Metal GPU.");
                                if ui
                                    .add(
                                        egui::Slider::new(
                                            &mut self.preview_frame_idx,
                                            0..=(num_frames - 1),
                                        )
                                        .text(format!("/ {} frames", num_frames)),
                                    )
                                    .changed()
                                {
                                    self.preview_dirty = true;
                                }
                            });
                        }
                    }
                }
            } else {
                // Empty Viewport State: Centered both vertically and horizontally
                let avail_rect = ui.available_rect_before_wrap();
                ui.painter().rect(
                    avail_rect,
                    Rounding::same(8.0),
                    if self.dark_mode {
                        Color32::from_rgb(12, 13, 17)
                    } else {
                        Color32::from_rgb(235, 237, 242)
                    },
                    Stroke::new(1.0_f32, Color32::from_rgb(42, 47, 58)),
                );

                let content_height = 205.0;
                let top_pad = ((avail_rect.height() - content_height) * 0.5).max(16.0);

                ui.allocate_new_ui(egui::UiBuilder::new().max_rect(avail_rect), |ui| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(top_pad);
                        if let Some(ref logo) = self.logo_texture {
                            ui.add(
                                egui::Image::new(logo)
                                    .fit_to_exact_size(egui::vec2(112.0, 112.0))
                                    .rounding(Rounding::same(22.0)),
                            );
                            ui.add_space(16.0);
                        }
                        ui.heading(
                            RichText::new("Nikon N-RAW (.nev) to 12-Bit ProRes 4444 / N-Log Studio")
                                .size(19.0)
                                .strong(),
                        );
                        ui.add_space(10.0);
                        ui.label(
                            RichText::new(
                                "Drag & drop a Nikon Z8 / Z9 .nev file anywhere onto this window,",
                            )
                            .size(13.5)
                            .weak(),
                        );
                        ui.add_space(2.0);
                        ui.label(
                            RichText::new(
                                "click '📂 Open .NEV File(s)...', or click '▶ Generate Sample .NEV Clip' in the top bar.",
                            )
                            .size(13.5)
                            .weak(),
                        );
                    });
                });
            }
        });
    }
}
