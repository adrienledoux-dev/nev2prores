use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Instant;

use crossbeam_channel::{unbounded, Receiver};
use eframe::egui::{self, Color32, ColorImage, RichText, TextureHandle, TextureOptions};

use crate::decoder::{NativeTicoRawDecoder, TicoRawDecoder};
use crate::demuxer::{demux_nev_file, NevDemuxedContainer};
use crate::encoder::ProResCodec;
use crate::metal_pipeline::{ColorPipelineParams, MetalColorPipeline, ScaleMode};
use crate::{
    generate_sample_nev_file, transcode_nev_file_with_progress, TranscodeOptions, TranscodeReport,
};

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

/// Launches the native macOS `nev2prores` Studio GUI window.
pub fn run_native_gui(
    initial_input: Option<PathBuf>,
    initial_output: Option<PathBuf>,
) -> anyhow::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(
                "nev2prores — Native macOS Nikon N-RAW (.nev) to 12-Bit ProRes 4444 / N-Log Studio",
            )
            .with_inner_size([1360.0, 860.0])
            .with_min_inner_size([1024.0, 680.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "nev2prores",
        native_options,
        Box::new(move |cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(Nev2ProResGuiApp::new(
                initial_input,
                initial_output,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("Failed to launch eframe native GUI: {}", e))
}

pub struct Nev2ProResGuiApp {
    gpu: Option<MetalColorPipeline>,
    gpu_name: String,
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
    pub fn new(initial_input: Option<PathBuf>, initial_output: Option<PathBuf>) -> Self {
        let (gpu, gpu_name) = match MetalColorPipeline::new() {
            Ok(p) => {
                let name = p.device_name();
                (Some(p), name)
            }
            Err(e) => (None, format!("Unavailable ({})", e)),
        };

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
                "Ready. Open a Nikon .nev file/folder or click 'Generate Sample .NEV Clip' to preview on the Metal GPU."
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
                    if !self.clips.iter().any(|c| c.path == path) {
                        self.clips.push(QueuedClip {
                            path: path.to_path_buf(),
                            container,
                        });
                    }
                    let new_idx = self.clips.len() - 1;
                    self.select_clip(new_idx, as_shot_k);
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

        // 2. Run Apple Metal Compute Shader (using Half scale if width > 1280 for >60fps interactive scrubbing)
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
        let apply_lut = self.display_rec709_lut;

        for chunk in rgba_f16.as_chunks::<4>().0 {
            let mut r = chunk[0].to_f32().clamp(0.0, 1.0);
            let mut g = chunk[1].to_f32().clamp(0.0, 1.0);
            let mut b = chunk[2].to_f32().clamp(0.0, 1.0);

            if apply_lut {
                // Soft S-curve display LUT transforming N-Log to Rec.709 monitoring contrast
                r = nlog_to_display_gamma(r);
                g = nlog_to_display_gamma(g);
                b = nlog_to_display_gamma(b);
            }

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
    // Simple filmic shoulder + Rec.709 1/2.2 display gamma
    let mapped = linear / (1.0 + linear * 0.35);
    mapped.powf(1.0 / 2.2).clamp(0.0, 1.0)
}

impl eframe::App for Nev2ProResGuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
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

        // Top Toolbar
        egui::TopBottomPanel::top("top_toolbar").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading(RichText::new("nev2prores Studio").strong());
                ui.separator();

                if ui.button("📂 Open .NEV File(s)...").clicked() {
                    if let Some(files) = rfd::FileDialog::new()
                        .add_filter("Nikon N-RAW (.nev)", &["nev", "NEV", "mov", "MOV"])
                        .pick_files()
                    {
                        for f in files {
                            self.add_path(&f);
                        }
                    }
                }

                if ui.button("📁 Open Folder...").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.add_path(&dir);
                    }
                }

                if ui.button("🧪 Generate Sample .NEV Clip").clicked() {
                    let sample_path = std::env::temp_dir().join("Nikon_Z9_Sample_1080p.NEV");
                    match generate_sample_nev_file(&sample_path) {
                        Ok(()) => self.add_path(&sample_path),
                        Err(e) => {
                            self.status_banner = format!("Sample generation error: {}", e);
                        }
                    }
                }

                ui.separator();
                ui.label(
                    RichText::new(format!("GPU: {}", self.gpu_name))
                        .color(Color32::from_rgb(120, 220, 160)),
                );
            });
            ui.add_space(4.0);
        });

        // Bottom Status & Progress Bar
        egui::TopBottomPanel::bottom("bottom_status").show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(&self.status_banner);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if let Some(last) = self.completed_reports.last() {
                        if ui.button("🎬 Open .MOV in QuickTime").clicked() {
                            let _ = Command::new("open").arg(&last.output_mov_path).spawn();
                        }
                        if ui.button("🔍 Reveal in Finder").clicked() {
                            let _ = Command::new("open")
                                .arg("-R")
                                .arg(&last.output_mov_path)
                                .spawn();
                        }
                    }
                });
            });
            if self.is_transcoding || self.progress_fraction > 0.0 {
                ui.add(
                    egui::ProgressBar::new(self.progress_fraction)
                        .text(&self.progress_text)
                        .animate(self.is_transcoding),
                );
            }
            ui.add_space(4.0);
        });

        // Left Panel: Clip Queue & Nikon NCDT MakerNote Inspector
        egui::SidePanel::left("left_inspector")
            .resizable(true)
            .default_width(320.0)
            .show(ctx, |ui| {
                ui.heading("Clip Queue");
                ui.separator();

                let mut clicked_idx = None;
                egui::ScrollArea::vertical()
                    .max_height(180.0)
                    .show(ui, |ui| {
                        if self.clips.is_empty() {
                            ui.weak("Drop .nev files here or click 'Generate Sample .NEV Clip'.");
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
                                "{} ({}x{}, {}f @ {:.2}fps)",
                                name,
                                clip.container.width,
                                clip.container.height,
                                clip.container.frames.len(),
                                fps
                            );
                            if ui.selectable_label(selected, label).clicked() {
                                clicked_idx =
                                    Some((idx, clip.container.metadata.color_temperature_kelvin));
                            }
                        }
                    });

                if let Some((idx, k)) = clicked_idx {
                    self.select_clip(idx, k);
                }

                ui.add_space(8.0);
                ui.heading("Nikon NCDT MakerNote");
                ui.separator();

                if let Some(idx) = self.selected_clip_idx {
                    if let Some(clip) = self.clips.get(idx) {
                        let c = &clip.container;
                        let (_, _, fps) = c.frame_rate();
                        egui::Grid::new("metadata_grid")
                            .num_columns(2)
                            .spacing([12.0, 4.0])
                            .striped(true)
                            .show(ui, |ui| {
                                ui.label("Camera:");
                                ui.label(&c.metadata.camera_model);
                                ui.end_row();

                                ui.label("Lens:");
                                ui.label(&c.metadata.lens_model);
                                ui.end_row();

                                ui.label("Sensor Format:");
                                ui.label(format!(
                                    "{}x{} ({}-bit RGGB)",
                                    c.width, c.height, c.bit_depth
                                ));
                                ui.end_row();

                                ui.label("Frame Rate:");
                                ui.label(format!("{:.3} fps ({} frames)", fps, c.frames.len()));
                                ui.end_row();

                                ui.label("ISO / Shutter:");
                                ui.label(format!(
                                    "ISO {}  |  {}",
                                    c.metadata.iso_speed,
                                    c.metadata.shutter_display()
                                ));
                                ui.end_row();

                                ui.label("Timecode (tmcd):");
                                ui.label(
                                    c.metadata
                                        .timecode
                                        .as_deref()
                                        .unwrap_or("None"),
                                );
                                ui.end_row();

                                ui.label("Audio Track:");
                                if let Some(ref a) = c.audio_track {
                                    ui.label(format!(
                                        "{}-bit PCM, {}ch @ {} Hz",
                                        a.sample_bits, a.channels, a.sample_rate
                                    ));
                                } else {
                                    ui.label("None");
                                }
                                ui.end_row();

                                ui.label("WB Gains (0x000C):");
                                ui.label(format!(
                                    "R:{:.3} B:{:.3} G:{:.2}",
                                    c.metadata.wb_rb_levels[0],
                                    c.metadata.wb_rb_levels[1],
                                    c.metadata.wb_rb_levels[2]
                                ));
                                ui.end_row();

                                ui.label("BlackLevel (0x003D):");
                                ui.label(format!(
                                    "{:?} / {}",
                                    c.metadata.black_level, c.metadata.white_level
                                ));
                                ui.end_row();
                            });

                        ui.add_space(10.0);
                        ui.label(RichText::new("Bradford Chromatic Adaptation M_WB:").strong());
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
                            "[{:+.4} {:+.4} {:+.4}]\n[{:+.4} {:+.4} {:+.4}]\n[{:+.4} {:+.4} {:+.4}]",
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
                    ui.weak("Select a clip to inspect its NCDT metadata.");
                }
            });

        // Right Panel: Scene-Linear RAW Controls & ProRes Export Settings
        egui::SidePanel::right("right_controls")
            .resizable(true)
            .default_width(340.0)
            .show(ctx, |ui| {
                ui.heading("Scene-Linear RAW Grading");
                ui.separator();

                if ui
                    .checkbox(
                        &mut self.override_wb,
                        "Override White Balance (Bradford CAT)",
                    )
                    .changed()
                {
                    self.preview_dirty = true;
                }

                ui.add_enabled_ui(self.override_wb, |ui| {
                    ui.label("Color Temperature (Kelvin):");
                    if ui
                        .add(
                            egui::Slider::new(&mut self.kelvin, 2000..=10000)
                                .suffix(" K")
                                .step_by(50.0),
                        )
                        .changed()
                    {
                        self.preview_dirty = true;
                    }

                    ui.horizontal(|ui| {
                        for &(label, k) in &[
                            ("3200K", 3200u32),
                            ("4300K", 4300),
                            ("5600K", 5600),
                            ("6500K", 6500),
                        ] {
                            if ui.small_button(label).clicked() {
                                self.kelvin = k;
                                self.preview_dirty = true;
                            }
                        }
                    });

                    ui.add_space(4.0);
                    ui.label("Green / Magenta Tint Shift:");
                    if ui
                        .add(egui::Slider::new(&mut self.tint, -50.0..=50.0).step_by(0.5))
                        .changed()
                    {
                        self.preview_dirty = true;
                    }
                });

                ui.add_space(10.0);
                ui.label(RichText::new("Scene-Linear Exposure (EV Stops):").strong());
                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::Slider::new(&mut self.ev_stops, -4.0..=4.0)
                                .suffix(" EV")
                                .step_by(0.05),
                        )
                        .changed()
                    {
                        self.preview_dirty = true;
                    }
                    if ui.small_button("0.0 EV").clicked() {
                        self.ev_stops = 0.0;
                        self.preview_dirty = true;
                    }
                });

                ui.add_space(16.0);
                ui.heading("ProRes & FCPXML Export");
                ui.separator();

                ui.label("Hardware VideoToolbox Codec:");
                ui.radio_value(
                    &mut self.codec,
                    ProResCodec::ProRes4444,
                    "Apple ProRes 4444 (ap4h, 12-bit 4:4:4)",
                );
                ui.radio_value(
                    &mut self.codec,
                    ProResCodec::ProRes422HQ,
                    "Apple ProRes 422 HQ (apch, 10-bit 4:2:2)",
                );

                ui.add_space(6.0);
                ui.label("Output Resolution Scale:");
                ui.horizontal(|ui| {
                    ui.radio_value(&mut self.scale, ScaleMode::Full, "Full (1:1 Native)");
                    ui.radio_value(&mut self.scale, ScaleMode::Half, "Half (1/2 Proxy)");
                });

                ui.add_space(6.0);
                ui.checkbox(
                    &mut self.emit_fcpxml,
                    "Generate Final Cut Pro FCPXML v1.11 (.fcpxml)",
                );

                ui.add_space(8.0);
                ui.label("Output Directory:");
                ui.monospace(self.output_dir.to_string_lossy());
                if ui.button("📁 Choose Output Folder...").clicked() {
                    if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                        self.output_dir = dir;
                    }
                }

                ui.add_space(16.0);
                ui.separator();

                let can_run = !self.is_transcoding && !self.clips.is_empty();
                ui.add_enabled_ui(can_run, |ui| {
                    if ui
                        .add_sized(
                            [ui.available_width(), 36.0],
                            egui::Button::new(
                                RichText::new("⚡ Transcode Selected Clip to ProRes").strong(),
                            ),
                        )
                        .clicked()
                    {
                        self.start_transcode_job(true);
                    }
                    ui.add_space(6.0);
                    if self.clips.len() > 1
                        && ui
                            .add_sized(
                                [ui.available_width(), 32.0],
                                egui::Button::new(format!(
                                    "⚡ Batch Transcode All {} Clips",
                                    self.clips.len()
                                )),
                            )
                            .clicked()
                    {
                        self.start_transcode_job(false);
                    }
                });
            });

        // Center Panel: Live Metal GPU Debayer & N-Log Viewport
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("Apple Metal GPU Viewport");
                ui.separator();
                if ui
                    .selectable_label(!self.display_rec709_lut, "Nikon N-Log (Encoded Signal)")
                    .clicked()
                {
                    self.display_rec709_lut = false;
                    self.preview_dirty = true;
                }
                if ui
                    .selectable_label(self.display_rec709_lut, "Rec. 709 Display LUT (Preview)")
                    .clicked()
                {
                    self.display_rec709_lut = true;
                    self.preview_dirty = true;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if self.preview_resolution.0 > 0 {
                        ui.label(
                            RichText::new(format!(
                                "Malvar-He-Cutler 5x5 + N-Log ({}x{}): {:.2} ms on GPU",
                                self.preview_resolution.0,
                                self.preview_resolution.1,
                                self.last_gpu_ms
                            ))
                            .monospace(),
                        );
                    }
                });
            });
            ui.separator();

            if let Some(ref tex) = self.preview_texture {
                let avail = ui.available_size() - egui::vec2(0.0, 36.0);
                let tex_size = tex.size_vec2();
                let aspect = tex_size.x / tex_size.y.max(1.0);
                let mut draw_w = avail.x;
                let mut draw_h = draw_w / aspect;
                if draw_h > avail.y {
                    draw_h = avail.y;
                    draw_w = draw_h * aspect;
                }

                ui.vertical_centered(|ui| {
                    ui.add(egui::Image::new(tex).fit_to_exact_size(egui::vec2(draw_w, draw_h)));
                });

                // Frame Scrubber at bottom of viewport
                if let Some(idx) = self.selected_clip_idx {
                    if let Some(clip) = self.clips.get(idx) {
                        let num_frames = clip.container.frames.len();
                        if num_frames > 1 {
                            ui.horizontal(|ui| {
                                ui.label("Frame Scrubber:");
                                if ui
                                    .add(
                                        egui::Slider::new(
                                            &mut self.preview_frame_idx,
                                            0..=(num_frames - 1),
                                        )
                                        .text("frame"),
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
                ui.centered_and_justified(|ui| {
                    ui.label(
                        RichText::new(
                            "No clip loaded.\nClick '🧪 Generate Sample .NEV Clip' or '📂 Open .NEV File(s)...' in the top bar.",
                        )
                        .size(16.0),
                    );
                });
            }
        });
    }
}
