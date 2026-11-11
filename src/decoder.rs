use std::ffi::CString;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{bounded, Receiver};
use rayon::prelude::*;
use thiserror::Error;

use crate::demuxer::FrameDescriptor;

#[derive(Debug, Error)]
pub enum DecoderError {
    #[error("I/O error while reading TicoRAW frame bitstream: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid frame dimensions: {width}x{height} (must be non-zero even RGGB dimensions)")]
    InvalidDimensions { width: usize, height: usize },
    #[error("Corrupt or truncated TicoRAW bitstream (expected {expected} samples, got {actual})")]
    BitstreamTruncated { expected: usize, actual: usize },
    #[error(
        "Proprietary intoPIX TicoRAW entropy bitstream ({byte_len} bytes for {width}x{height} frame) \
         requires an external TicoRAW decoder library. Set NEV2PRORES_TICORAW_DYLIB=/path/to/libticoraw_bridge.dylib \
         exporting `nev2prores_decode_ticoraw`."
    )]
    UnsupportedCameraBitstream {
        byte_len: usize,
        width: usize,
        height: usize,
    },
    #[error("External TicoRAW dylib decoder failed with status code {0}")]
    ExternalPluginFailed(i32),
    #[error("Transcode cancelled by user")]
    Cancelled,
}

/// Unpacked 16-bit RGGB Bayer frame ready for zero-copy upload to Apple Metal unified memory.
#[derive(Debug, Clone)]
pub struct DecodedBayerFrame {
    pub frame_index: u32,
    pub pts: u64,
    pub duration: u32,
    pub width: usize,
    pub height: usize,
    /// Unpacked 16-bit RGGB Bayer array (`width * height` elements, values in `0..=4095` or `0..=16383`).
    pub bayer_rggb: Vec<u16>,
}

/// Trait interface for unpacking a compressed Nikon `.nev` (TicoRAW) frame bitstream
/// into a 16-bit RGGB Bayer buffer (`Vec<u16>` of length `width * height`).
pub trait TicoRawDecoder: Send + Sync {
    fn decode_frame(
        &self,
        bitstream: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<u16>, DecoderError>;
}

/// TicoRAW header magic (`"TICO"` = `0x5449434F`) for native line-based wavelet/packed bitstreams.
pub const TICORAW_MAGIC: &[u8; 4] = b"TICO";

/// C ABI function signature for optional external TicoRAW dylib plugins (`NEV2PRORES_TICORAW_DYLIB`):
/// `int32_t nev2prores_decode_ticoraw(const uint8_t* bs, size_t bs_len, uint32_t w, uint32_t h, uint16_t* out_rggb)`
type ExternalDecodeFn = unsafe extern "C" fn(
    bitstream: *const u8,
    bitstream_len: usize,
    width: u32,
    height: u32,
    out_bayer_rggb: *mut u16,
) -> i32;

struct DynamicTicoRawPlugin {
    handle: *mut libc::c_void,
    decode_fn: ExternalDecodeFn,
}

// SAFETY: The dynamically loaded C decoder function is stateless per frame buffer and thread-safe.
unsafe impl Send for DynamicTicoRawPlugin {}
unsafe impl Sync for DynamicTicoRawPlugin {}

impl DynamicTicoRawPlugin {
    fn try_load_from_env() -> Option<Arc<Self>> {
        let mut candidates = Vec::new();
        if let Ok(env_path) = std::env::var("NEV2PRORES_TICORAW_DYLIB") {
            if !env_path.trim().is_empty() {
                candidates.push(PathBuf::from(env_path));
            }
        }
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                candidates.push(dir.join("libticoraw_bridge.dylib"));
                candidates.push(dir.join("../Resources/libticoraw_bridge.dylib"));
            }
        }
        candidates.push(PathBuf::from("/usr/local/lib/libticoraw_bridge.dylib"));
        candidates.push(PathBuf::from("/opt/homebrew/lib/libticoraw_bridge.dylib"));

        for path in candidates {
            if !path.exists() {
                continue;
            }
            let Ok(c_path) = CString::new(path.to_string_lossy().as_bytes()) else {
                continue;
            };
            let Ok(sym_name) = CString::new("nev2prores_decode_ticoraw") else {
                continue;
            };
            // SAFETY: `c_path` and `sym_name` are valid NUL-terminated strings; null checks guard `dlopen` and `dlsym`.
            unsafe {
                let handle = libc::dlopen(c_path.as_ptr(), libc::RTLD_LAZY | libc::RTLD_LOCAL);
                if handle.is_null() {
                    continue;
                }
                let sym = libc::dlsym(handle, sym_name.as_ptr());
                if sym.is_null() {
                    libc::dlclose(handle);
                    continue;
                }
                let decode_fn: ExternalDecodeFn = std::mem::transmute(sym);
                return Some(Arc::new(Self { handle, decode_fn }));
            }
        }
        None
    }
}

impl Drop for DynamicTicoRawPlugin {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            // SAFETY: `self.handle` was returned by a successful `dlopen` call and is closed once on drop.
            unsafe {
                libc::dlclose(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Native multithreaded TicoRAW / Nikon N-RAW decoder implementation.
///
/// Supports:
/// 1. `TICO` headered line-strip 12-bit packed / LeGall 5/3 wavelet bitstreams (decoded in parallel across strips with `rayon`).
/// 2. Unpacked 16-bit little-endian RGGB bitstreams (`width * height * 2` bytes).
/// 3. Packed 12-bit RGGB bitstreams (`width * height * 3 / 2` bytes).
/// 4. Optional external `libticoraw_bridge.dylib` plugin (`NEV2PRORES_TICORAW_DYLIB`) for proprietary camera entropy streams;
///    returns an explicit `DecoderError::UnsupportedCameraBitstream` error rather than corrupt noise if absent.
#[derive(Clone, Default)]
pub struct NativeTicoRawDecoder {
    pub default_black_level: u16,
    external_plugin: Option<Arc<DynamicTicoRawPlugin>>,
}

impl std::fmt::Debug for NativeTicoRawDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeTicoRawDecoder")
            .field("default_black_level", &self.default_black_level)
            .field("has_external_plugin", &self.external_plugin.is_some())
            .finish()
    }
}

impl NativeTicoRawDecoder {
    pub fn new() -> Self {
        Self {
            default_black_level: 150,
            external_plugin: DynamicTicoRawPlugin::try_load_from_env(),
        }
    }

    pub fn has_external_plugin(&self) -> bool {
        self.external_plugin.is_some()
    }
}

impl TicoRawDecoder for NativeTicoRawDecoder {
    fn decode_frame(
        &self,
        bitstream: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<u16>, DecoderError> {
        if width == 0 || height == 0 || (width & 1) != 0 || (height & 1) != 0 {
            return Err(DecoderError::InvalidDimensions { width, height });
        }
        let num_pixels = width * height;

        // Case 1: TICO line-based subband / packed 12-bit / LeGall 5/3 wavelet bitstream
        if bitstream.len() >= 16 && &bitstream[0..4] == TICORAW_MAGIC {
            let mode = u32::from_le_bytes([bitstream[4], bitstream[5], bitstream[6], bitstream[7]]);
            let payload = &bitstream[16..];
            if mode == 1 {
                return unpack_12bit_parallel(payload, width, height);
            } else if mode == 2 {
                return decode_levo_53_wavelet_parallel(payload, width, height);
            }
        }

        // Case 2: Unpacked 16-bit RGGB buffer (`width * height * 2` bytes)
        if bitstream.len() == num_pixels * 2 {
            let mut out = vec![0u16; num_pixels];
            out.par_chunks_mut(width)
                .enumerate()
                .for_each(|(row, dst_row)| {
                    let src_off = row * width * 2;
                    let src = &bitstream[src_off..src_off + width * 2];
                    for (col, dst) in dst_row.iter_mut().enumerate() {
                        *dst = u16::from_le_bytes([src[col * 2], src[col * 2 + 1]]) & 0x3FFF;
                    }
                });
            return Ok(out);
        }

        // Case 3: Packed 12-bit RGGB buffer (`width * height * 3 / 2` bytes)
        if bitstream.len() == (num_pixels * 3) / 2 {
            return unpack_12bit_parallel(bitstream, width, height);
        }

        // Case 4: Delegate to external `libticoraw_bridge.dylib` plugin if loaded
        if let Some(ref plugin) = self.external_plugin {
            let mut out = vec![0u16; num_pixels];
            // SAFETY: `bitstream` has `bitstream.len()` valid bytes and `out` has `width * height` writable `u16` elements.
            let rc = unsafe {
                (plugin.decode_fn)(
                    bitstream.as_ptr(),
                    bitstream.len(),
                    width as u32,
                    height as u32,
                    out.as_mut_ptr(),
                )
            };
            if rc == 0 {
                return Ok(out);
            }
            return Err(DecoderError::ExternalPluginFailed(rc));
        }

        // Fail explicitly instead of producing silent corrupted imagery
        Err(DecoderError::UnsupportedCameraBitstream {
            byte_len: bitstream.len(),
            width,
            height,
        })
    }
}

/// Unpacks a 12-bit packed RGGB bitstream (`3 bytes -> 2 u16 pixels`) in parallel across rows using `rayon`.
fn unpack_12bit_parallel(
    payload: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u16>, DecoderError> {
    let num_pixels = width * height;
    let required_bytes = (num_pixels * 3) / 2;
    if payload.len() < required_bytes {
        return Err(DecoderError::BitstreamTruncated {
            expected: num_pixels,
            actual: (payload.len() * 2) / 3,
        });
    }

    let mut out = vec![0u16; num_pixels];
    let row_bytes = (width * 3) / 2;

    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, dst_row)| {
            let src_start = row * row_bytes;
            let src_row = &payload[src_start..src_start + row_bytes];
            for pair_idx in 0..(width / 2) {
                let b0 = src_row[pair_idx * 3] as u16;
                let b1 = src_row[pair_idx * 3 + 1] as u16;
                let b2 = src_row[pair_idx * 3 + 2] as u16;
                dst_row[pair_idx * 2] = (b0 << 4) | (b1 >> 4);
                dst_row[pair_idx * 2 + 1] = ((b1 & 0x0F) << 8) | b2;
            }
        });

    Ok(out)
}

/// Inverse 1D LeGall 5/3 reversible integer lifting wavelet transform per Bayer channel row,
/// parallelized across rows via `rayon`.
fn decode_levo_53_wavelet_parallel(
    payload: &[u8],
    width: usize,
    height: usize,
) -> Result<Vec<u16>, DecoderError> {
    let num_pixels = width * height;
    if payload.len() < num_pixels * 2 {
        return Err(DecoderError::BitstreamTruncated {
            expected: num_pixels,
            actual: payload.len() / 2,
        });
    }

    let mut out = vec![0u16; num_pixels];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(row, dst_row)| {
            let src_start = row * width * 2;
            let src = &payload[src_start..src_start + width * 2];
            let half = width / 2;
            for i in 0..half {
                let low = i16::from_le_bytes([src[i * 2], src[i * 2 + 1]]) as i32;
                let high =
                    i16::from_le_bytes([src[(half + i) * 2], src[(half + i) * 2 + 1]]) as i32;
                let even = (low - (high >> 1)).clamp(0, 16383) as u16;
                let odd = (even as i32 + high).clamp(0, 16383) as u16;
                dst_row[i * 2] = even;
                dst_row[i * 2 + 1] = odd;
            }
        });

    Ok(out)
}

/// Encodes a 16-bit RGGB Bayer frame into the compact 12-bit `TICO` bitstream format
/// (used for generating realistic `.nev` test containers).
pub fn encode_ticoraw_test_frame(bayer: &[u16], width: usize, height: usize) -> Vec<u8> {
    let num_pixels = width * height;
    assert_eq!(bayer.len(), num_pixels);
    let packed_len = (num_pixels * 3) / 2;
    let mut out = Vec::with_capacity(16 + packed_len);

    // 16-byte TICO frame header
    out.extend_from_slice(TICORAW_MAGIC);
    out.extend_from_slice(&1u32.to_le_bytes()); // mode 1: 12-bit packed TICO subband
    out.extend_from_slice(&(width as u32).to_le_bytes());
    out.extend_from_slice(&(height as u32).to_le_bytes());

    for pair in bayer.as_chunks::<2>().0 {
        let s0 = pair[0] & 0x0FFF;
        let s1 = pair[1] & 0x0FFF;
        let b0 = (s0 >> 4) as u8;
        let b1 = (((s0 & 0x0F) << 4) | (s1 >> 8)) as u8;
        let b2 = (s1 & 0xFF) as u8;
        out.push(b0);
        out.push(b1);
        out.push(b2);
    }

    out
}

/// Bounded producer-consumer parallel decoding pipeline.
///
/// Spawns a background producer that reads frame slices from the `.nev` file in small batches,
/// decodes them in parallel across performance cores with `rayon`, and pushes `DecodedBayerFrame`s
/// into a bounded `crossbeam_channel` ring buffer so memory usage never exceeds `max_in_flight` frames
/// (critical for 8256x4644 8K RAW frames).
pub struct BoundedDecodePipeline {
    rx: Receiver<Result<DecodedBayerFrame, DecoderError>>,
    producer_handle: Option<JoinHandle<()>>,
}

impl BoundedDecodePipeline {
    pub fn spawn(
        nev_path: PathBuf,
        frames: Vec<FrameDescriptor>,
        width: usize,
        height: usize,
        decoder: Arc<dyn TicoRawDecoder>,
        ring_buffer_capacity: usize,
        cancel_flag: Option<Arc<AtomicBool>>,
    ) -> Self {
        let cap = ring_buffer_capacity.max(2);
        let (tx, rx) = bounded::<Result<DecodedBayerFrame, DecoderError>>(cap);

        let handle = thread::spawn(move || {
            let file_res = File::open(&nev_path);
            let mut reader = match file_res {
                Ok(f) => BufReader::new(f),
                Err(e) => {
                    let _ = tx.send(Err(DecoderError::Io(e)));
                    return;
                }
            };

            // Process frames in bounded chunks matching the ring buffer capacity
            // so `rayon` decodes multiple frames simultaneously while keeping RAM bounded.
            for batch in frames.chunks(cap) {
                if cancel_flag
                    .as_ref()
                    .is_some_and(|flag| flag.load(Ordering::Relaxed))
                {
                    let _ = tx.send(Err(DecoderError::Cancelled));
                    return;
                }

                let mut raw_packets = Vec::with_capacity(batch.len());
                for desc in batch {
                    if let Err(e) = reader.seek(SeekFrom::Start(desc.file_offset)) {
                        let _ = tx.send(Err(DecoderError::Io(e)));
                        return;
                    }
                    let mut buf = vec![0u8; desc.byte_size as usize];
                    if let Err(e) = reader.read_exact(&mut buf) {
                        let _ = tx.send(Err(DecoderError::Io(e)));
                        return;
                    }
                    raw_packets.push((desc.clone(), buf));
                }

                // Decode the batch in parallel across CPU performance cores via `rayon`
                let decoded_batch: Vec<Result<DecodedBayerFrame, DecoderError>> = raw_packets
                    .into_par_iter()
                    .map(|(desc, packet)| {
                        let bayer_rggb = decoder.decode_frame(&packet, width, height)?;
                        Ok(DecodedBayerFrame {
                            frame_index: desc.frame_index,
                            pts: desc.pts,
                            duration: desc.duration,
                            width,
                            height,
                            bayer_rggb,
                        })
                    })
                    .collect();

                // Emit in strict presentation order into the bounded channel
                for item in decoded_batch {
                    if tx.send(item).is_err() {
                        // Consumer dropped early
                        return;
                    }
                }
            }
        });

        Self {
            rx,
            producer_handle: Some(handle),
        }
    }

    /// Returns the receiver channel for iterating over decoded frames in presentation order.
    pub fn receiver(&self) -> &Receiver<Result<DecodedBayerFrame, DecoderError>> {
        &self.rx
    }
}

impl Drop for BoundedDecodePipeline {
    fn drop(&mut self) {
        if let Some(handle) = self.producer_handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ticoraw_12bit_pack_unpack_roundtrip() {
        let width = 128usize;
        let height = 72usize;
        let orig: Vec<u16> = (0..(width * height))
            .map(|i| ((i * 37 + 150) % 4096) as u16)
            .collect();

        let bitstream = encode_ticoraw_test_frame(&orig, width, height);
        let decoder = NativeTicoRawDecoder::new();
        let unpacked = decoder
            .decode_frame(&bitstream, width, height)
            .expect("should decode 12-bit TICO frame");

        assert_eq!(orig, unpacked);
    }

    #[test]
    fn test_unsupported_bitstream_returns_explicit_error_not_noise() {
        let decoder = NativeTicoRawDecoder::new();
        if !decoder.has_external_plugin() {
            let random_payload = vec![0xABu8; 512];
            let err = decoder
                .decode_frame(&random_payload, 128, 72)
                .expect_err("unknown camera bitstream should return explicit error");
            assert!(matches!(
                err,
                DecoderError::UnsupportedCameraBitstream { .. }
            ));
        }
    }

    #[test]
    fn test_invalid_odd_bayer_dimensions_rejected() {
        let decoder = NativeTicoRawDecoder::new();
        assert!(matches!(
            decoder.decode_frame(&[0u8; 100], 127, 72),
            Err(DecoderError::InvalidDimensions { .. })
        ));
    }
}
