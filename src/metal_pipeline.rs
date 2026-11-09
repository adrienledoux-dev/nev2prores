use std::ffi::c_void;
use std::ptr;

use bytemuck::{Pod, Zeroable};
use half::f16;
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use objc::rc::autoreleasepool;
use thiserror::Error;

use crate::metadata::{compute_wb_color_matrix, NikonMetadata};

/// Embedded Metal compute shader source (`shaders/pipeline.metal`).
pub const PIPELINE_METAL_SOURCE: &str = include_str!("../shaders/pipeline.metal");

#[derive(Debug, Error)]
pub enum MetalPipelineError {
    #[error("No Apple Metal GPU device found on this system")]
    NoMetalDevice,
    #[error("Failed to compile Metal compute shader 'shaders/pipeline.metal': {0}")]
    ShaderCompilation(String),
    #[error("Failed to create Metal compute pipeline state: {0}")]
    PipelineState(String),
    #[error("Input Bayer buffer size mismatch: expected {expected} u16 elements, got {actual}")]
    InputBufferSizeMismatch { expected: usize, actual: usize },
}

/// Output resolution scale mode (`--scale <full|half>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleMode {
    Full,
    Half,
}

impl ScaleMode {
    #[inline]
    pub fn factor(self) -> u32 {
        match self {
            ScaleMode::Full => 1,
            ScaleMode::Half => 2,
        }
    }

    #[inline]
    pub fn output_dimensions(self, width: u32, height: u32) -> (u32, u32) {
        let f = self.factor();
        // Ensure even dimensions for ProRes macroblocks
        let ow = ((width / f) & !1).max(2);
        let oh = ((height / f) & !1).max(2);
        (ow, oh)
    }
}

/// High-level scene-linear RAW processing parameters for a frame or clip.
#[derive(Debug, Clone)]
pub struct ColorPipelineParams {
    pub width: u32,
    pub height: u32,
    pub scale: ScaleMode,
    pub black_level: [u16; 4],
    pub white_level: u16,
    pub wb_gains_rggb: [f32; 4],
    pub as_shot_kelvin: u32,
    pub target_kelvin: Option<u32>,
    pub target_tint: Option<f32>,
    pub ev_stops: f32,
}

impl ColorPipelineParams {
    pub fn from_metadata(
        width: u32,
        height: u32,
        meta: &NikonMetadata,
        scale: ScaleMode,
        target_kelvin: Option<u32>,
        target_tint: Option<f32>,
        ev_stops: Option<f32>,
    ) -> Self {
        Self {
            width,
            height,
            scale,
            black_level: meta.black_level,
            white_level: meta.white_level,
            wb_gains_rggb: meta.rggb_wb_gains(),
            as_shot_kelvin: meta.color_temperature_kelvin,
            target_kelvin,
            target_tint,
            ev_stops: ev_stops.unwrap_or(0.0),
        }
    }

    pub fn build_uniforms(&self, out_stride_elements: u32) -> ColorPipelineUniforms {
        let (out_width, out_height) = self.scale.output_dimensions(self.width, self.height);
        let ev_scale = 2.0f32.powf(self.ev_stops);
        let m_wb =
            compute_wb_color_matrix(self.as_shot_kelvin, self.target_kelvin, self.target_tint);

        ColorPipelineUniforms {
            width: self.width,
            height: self.height,
            out_width,
            out_height,
            out_stride_elements,
            scale_factor: self.scale.factor(),
            white_level: self.white_level as f32,
            ev_scale,
            black_level: [
                self.black_level[0] as f32,
                self.black_level[1] as f32,
                self.black_level[2] as f32,
                self.black_level[3] as f32,
            ],
            wb_gains: self.wb_gains_rggb,
            color_matrix_row0: [m_wb[0][0], m_wb[0][1], m_wb[0][2], 0.0],
            color_matrix_row1: [m_wb[1][0], m_wb[1][1], m_wb[1][2], 0.0],
            color_matrix_row2: [m_wb[2][0], m_wb[2][1], m_wb[2][2], 0.0],
        }
    }
}

/// GPU Uniform buffer layout (`#[repr(C)]`), identical to `ColorPipelineUniforms` in `shaders/pipeline.metal`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ColorPipelineUniforms {
    pub width: u32,
    pub height: u32,
    pub out_width: u32,
    pub out_height: u32,
    pub out_stride_elements: u32,
    pub scale_factor: u32,
    pub white_level: f32,
    pub ev_scale: f32,
    pub black_level: [f32; 4],
    pub wb_gains: [f32; 4],
    pub color_matrix_row0: [f32; 4],
    pub color_matrix_row1: [f32; 4],
    pub color_matrix_row2: [f32; 4],
}

/// Apple Metal GPU Compute Pipeline for Malvar-He-Cutler 5x5 Demosaicing,
/// Scene-Linear RAW White Balance / Exposure / Bradford Adaptation, and Nikon N-Log Encoding.
pub struct MetalColorPipeline {
    device: Device,
    command_queue: CommandQueue,
    pso_rgba16f: ComputePipelineState,
    pso_ayuv16: ComputePipelineState,
    cached_bayer_buffer: Option<(usize, Buffer)>,
    cached_output_buffer: Option<(usize, Buffer)>,
}

impl MetalColorPipeline {
    /// Initializes the Metal compute pipeline on the default Apple Silicon GPU.
    pub fn new() -> Result<Self, MetalPipelineError> {
        let device = Device::system_default().ok_or(MetalPipelineError::NoMetalDevice)?;
        let command_queue = device.new_command_queue();

        let compile_opts = CompileOptions::new();
        let library = device
            .new_library_with_source(PIPELINE_METAL_SOURCE, &compile_opts)
            .map_err(MetalPipelineError::ShaderCompilation)?;

        let fn_rgba16f = library
            .get_function("debayer_nlog_rgba16f", None)
            .map_err(MetalPipelineError::PipelineState)?;
        let pso_rgba16f = device
            .new_compute_pipeline_state_with_function(&fn_rgba16f)
            .map_err(MetalPipelineError::PipelineState)?;

        let fn_ayuv16 = library
            .get_function("debayer_nlog_ayuv16", None)
            .map_err(MetalPipelineError::PipelineState)?;
        let pso_ayuv16 = device
            .new_compute_pipeline_state_with_function(&fn_ayuv16)
            .map_err(MetalPipelineError::PipelineState)?;

        Ok(Self {
            device,
            command_queue,
            pso_rgba16f,
            pso_ayuv16,
            cached_bayer_buffer: None,
            cached_output_buffer: None,
        })
    }

    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Ensures we have a reusable `StorageModeShared` unified memory buffer of `byte_len` bytes
    /// and copies the unpacked `u16` RGGB Bayer slice into it.
    fn upload_bayer_shared(&mut self, bayer_rggb: &[u16]) -> &Buffer {
        let byte_len = std::mem::size_of_val(bayer_rggb);
        let needs_alloc = match &self.cached_bayer_buffer {
            Some((cap, _)) => *cap < byte_len,
            None => true,
        };
        if needs_alloc {
            let buf = self
                .device
                .new_buffer(byte_len as u64, MTLResourceOptions::StorageModeShared);
            self.cached_bayer_buffer = Some((byte_len, buf));
        }

        let (_, ref buf) = self.cached_bayer_buffer.as_ref().unwrap();
        unsafe {
            ptr::copy_nonoverlapping(
                bayer_rggb.as_ptr() as *const u8,
                buf.contents() as *mut u8,
                byte_len,
            );
        }
        buf
    }

    /// Dispatches the Metal compute shader directly into a locked `CVPixelBuffer` base address
    /// (`kCVPixelFormatType_64RGBAHalf` or `kCVPixelFormatType_4444AYpCbCr16`).
    ///
    /// On Apple Silicon Unified Memory, IOSurface-backed `CVPixelBuffer` memory is page-aligned
    /// (`16384` bytes), allowing `new_buffer_with_bytes_no_copy` with `StorageModeShared` for true
    /// zero-copy GPU writing directly into the `VideoToolbox` pixel buffer!
    pub fn process_into_pixel_buffer_memory(
        &mut self,
        bayer_rggb: &[u16],
        params: &ColorPipelineParams,
        dst_base_addr: *mut c_void,
        dst_bytes_per_row: usize,
        dst_total_bytes: usize,
        use_ayuv16: bool,
    ) -> Result<(), MetalPipelineError> {
        let expected_samples = (params.width as usize) * (params.height as usize);
        if bayer_rggb.len() != expected_samples {
            return Err(MetalPipelineError::InputBufferSizeMismatch {
                expected: expected_samples,
                actual: bayer_rggb.len(),
            });
        }

        // Each pixel in `64RGBAHalf` or `4444AYpCbCr16` is 8 bytes (`half4` or `ushort4`).
        let out_stride_elements = (dst_bytes_per_row / 8) as u32;
        let uniforms = params.build_uniforms(out_stride_elements);

        autoreleasepool(|| {
            self.upload_bayer_shared(bayer_rggb);

            // Check if BOTH destination pointer and byte length are page-aligned (4096 bytes)
            // for safe zero-copy MTLBuffer wrapping without mapping past the allocation boundary.
            const PAGE_MASK: usize = 4095;
            let addr_aligned = (dst_base_addr as usize & PAGE_MASK) == 0;
            let size_aligned = (dst_total_bytes & PAGE_MASK) == 0;

            let no_copy_buf = if addr_aligned && size_aligned && dst_total_bytes > 0 {
                Some(self.device.new_buffer_with_bytes_no_copy(
                    dst_base_addr,
                    dst_total_bytes as u64,
                    MTLResourceOptions::StorageModeShared,
                    None,
                ))
            } else {
                let needs_alloc = match &self.cached_output_buffer {
                    Some((cap, _)) => *cap < dst_total_bytes,
                    None => true,
                };
                if needs_alloc {
                    let buf = self.device.new_buffer(
                        dst_total_bytes.max(8) as u64,
                        MTLResourceOptions::StorageModeShared,
                    );
                    self.cached_output_buffer = Some((dst_total_bytes.max(8), buf));
                }
                None
            };

            let (_, ref bayer_buf) = self.cached_bayer_buffer.as_ref().unwrap();
            let out_buf_ref: &Buffer = match no_copy_buf {
                Some(ref b) => b,
                None => &self.cached_output_buffer.as_ref().unwrap().1,
            };

            let cmd_buf = self.command_queue.new_command_buffer();
            let encoder = cmd_buf.new_compute_command_encoder();

            let pso = if use_ayuv16 {
                &self.pso_ayuv16
            } else {
                &self.pso_rgba16f
            };
            encoder.set_compute_pipeline_state(pso);
            encoder.set_buffer(0, Some(bayer_buf), 0);
            encoder.set_buffer(1, Some(out_buf_ref), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<ColorPipelineUniforms>() as u64,
                bytemuck::bytes_of(&uniforms).as_ptr() as *const c_void,
            );

            let tg_w = 16u64;
            let tg_h = 16u64;
            let threadgroup_size = MTLSize::new(tg_w, tg_h, 1);
            let grid_groups = MTLSize::new(
                (uniforms.out_width as u64).div_ceil(tg_w),
                (uniforms.out_height as u64).div_ceil(tg_h),
                1,
            );

            encoder.dispatch_thread_groups(grid_groups, threadgroup_size);
            encoder.end_encoding();

            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            if no_copy_buf.is_none() {
                unsafe {
                    ptr::copy_nonoverlapping(
                        out_buf_ref.contents() as *const u8,
                        dst_base_addr as *mut u8,
                        dst_total_bytes,
                    );
                }
            }

            Ok(())
        })
    }

    /// Processes an unpacked 16-bit RGGB Bayer buffer on the GPU and returns a tightly packed
    /// `Vec<f16>` (`4 * out_width * out_height` RGBAHalf values).
    pub fn process_frame_to_rgba16f_vec(
        &mut self,
        bayer_rggb: &[u16],
        params: &ColorPipelineParams,
    ) -> Result<Vec<f16>, MetalPipelineError> {
        let (out_w, out_h) = params.scale.output_dimensions(params.width, params.height);
        let num_floats = (out_w as usize) * (out_h as usize) * 4;
        let mut output = vec![f16::ZERO; num_floats];
        let bytes_per_row = (out_w as usize) * 8;
        let total_bytes = num_floats * 2;

        self.process_into_pixel_buffer_memory(
            bayer_rggb,
            params,
            output.as_mut_ptr() as *mut c_void,
            bytes_per_row,
            total_bytes,
            false,
        )?;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::nikon_nlog_encode;

    #[test]
    fn test_metal_gpu_debayer_and_nlog_shader_execution() {
        let mut gpu = MetalColorPipeline::new().expect("Apple Metal GPU should be available");
        let width = 64u32;
        let height = 36u32;

        // Create a flat 18% gray neutral sensor field so Malvar-He-Cutler demosaic has zero spatial gradients.
        // With black_level = 150, white_level = 4095, 18% linear value corresponds to:
        // raw_adu = 150 + round(0.18 * (4095 - 150)) = 150 + 710 = 860
        let raw_adu = 860u16;
        let bayer = vec![raw_adu; (width * height) as usize];

        let params = ColorPipelineParams {
            width,
            height,
            scale: ScaleMode::Full,
            black_level: [150, 150, 150, 150],
            white_level: 4095,
            wb_gains_rggb: [1.0, 1.0, 1.0, 1.0],
            as_shot_kelvin: 5600,
            target_kelvin: Some(5600),
            target_tint: Some(0.0),
            ev_stops: 0.0,
        };

        let rgba_f16 = gpu
            .process_frame_to_rgba16f_vec(&bayer, &params)
            .expect("GPU shader should execute");

        assert_eq!(rgba_f16.len(), (width * height * 4) as usize);

        // Inspect center pixel (32, 18)
        let idx = ((18 * width + 32) * 4) as usize;
        let r = rgba_f16[idx].to_f32();
        let g = rgba_f16[idx + 1].to_f32();
        let b = rgba_f16[idx + 2].to_f32();
        let a = rgba_f16[idx + 3].to_f32();

        let expected_linear = (raw_adu as f32 - 150.0) / (4095.0 - 150.0);
        let expected_nlog = nikon_nlog_encode(expected_linear);

        assert!(
            (g - expected_nlog).abs() < 0.02,
            "g={} expected={}",
            g,
            expected_nlog
        );
        assert!(
            (r - expected_nlog).abs() < 0.03,
            "r={} expected={}",
            r,
            expected_nlog
        );
        assert!(
            (b - expected_nlog).abs() < 0.03,
            "b={} expected={}",
            b,
            expected_nlog
        );
        assert!((a - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_metal_gpu_half_scale_and_ev_exposure() {
        let mut gpu = MetalColorPipeline::new().expect("Apple Metal GPU should be available");
        let width = 64u32;
        let height = 36u32;
        let bayer = vec![860u16; (width * height) as usize];

        let mut params = ColorPipelineParams {
            width,
            height,
            scale: ScaleMode::Half,
            black_level: [150, 150, 150, 150],
            white_level: 4095,
            wb_gains_rggb: [1.0, 1.0, 1.0, 1.0],
            as_shot_kelvin: 5600,
            target_kelvin: Some(5600),
            target_tint: Some(0.0),
            ev_stops: 0.0,
        };

        let out_0ev = gpu.process_frame_to_rgba16f_vec(&bayer, &params).unwrap();
        assert_eq!(out_0ev.len(), ((width / 2) * (height / 2) * 4) as usize);

        params.ev_stops = 1.0; // +1 stop doubles scene-linear light prior to N-Log encoding
        let out_plus1ev = gpu.process_frame_to_rgba16f_vec(&bayer, &params).unwrap();

        let g0 = out_0ev[1].to_f32();
        let g1 = out_plus1ev[1].to_f32();
        assert!(
            g1 > g0 + 0.08,
            "+1 EV should increase N-Log output ({} vs {})",
            g1,
            g0
        );
    }
}
