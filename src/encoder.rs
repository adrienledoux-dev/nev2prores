use std::ffi::c_void;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::ptr;
use std::sync::Mutex;

use core_foundation_sys::base::{
    kCFAllocatorDefault, CFAllocatorRef, CFRelease, CFTypeRef, OSStatus,
};
use core_foundation_sys::dictionary::CFDictionaryRef;
use core_foundation_sys::number::kCFBooleanFalse;
use core_foundation_sys::string::CFStringRef;
use thiserror::Error;

use crate::demuxer::{AudioTrackInfo, TimecodeTrackInfo};
use crate::metal_pipeline::{ColorPipelineParams, MetalColorPipeline, MetalPipelineError};

#[derive(Debug, Error)]
pub enum EncoderError {
    #[error("I/O error while writing .mov container: {0}")]
    Io(#[from] std::io::Error),
    #[error("Apple VideoToolbox error (OSStatus = {0}) during {1}")]
    VideoToolbox(i32, &'static str),
    #[error("CoreVideo pixel buffer allocation failed (CVReturn = {0})")]
    CoreVideo(i32),
    #[error("Metal GPU pipeline error during frame processing: {0}")]
    Metal(#[from] MetalPipelineError),
}

/// Target hardware Apple ProRes codec variant (`--codec <prores4444|prores422hq>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProResCodec {
    /// Apple ProRes 4444 (`ap4h`, 12-bit 4:4:4)
    ProRes4444,
    /// Apple ProRes 422 HQ (`apch`, 10-bit 4:2:2)
    ProRes422HQ,
}

impl ProResCodec {
    #[inline]
    pub fn fourcc(self) -> [u8; 4] {
        match self {
            ProResCodec::ProRes4444 => *b"ap4h",
            ProResCodec::ProRes422HQ => *b"apch",
        }
    }

    #[inline]
    pub fn cm_codec_type(self) -> u32 {
        u32::from_be_bytes(self.fourcc())
    }

    #[inline]
    pub fn display_name(self) -> &'static str {
        match self {
            ProResCodec::ProRes4444 => "Apple ProRes 4444 (ap4h, 12-bit 4:4:4)",
            ProResCodec::ProRes422HQ => "Apple ProRes 422 HQ (apch, 10-bit 4:2:2)",
        }
    }
}

// ============================================================================
// Apple CoreMedia / CoreVideo / VideoToolbox C FFI Declarations
// ============================================================================

pub type CVReturn = i32;
pub type CVImageBufferRef = *mut c_void;
pub type CVPixelBufferRef = CVImageBufferRef;
pub type CVPixelBufferPoolRef = *mut c_void;
pub type VTCompressionSessionRef = *mut c_void;
pub type CMSampleBufferRef = *mut c_void;
pub type CMBlockBufferRef = *mut c_void;

/// `kCVPixelFormatType_64RGBAHalf` (`'RGhA'`, 64-bit IEEE half-float RGBA)
pub const K_CV_PIXEL_FORMAT_TYPE_64_RGBA_HALF: u32 = u32::from_be_bytes(*b"RGhA");
/// `kCVPixelFormatType_4444AYpCbCr16` (`'y416'`, 16-bit 4:4:4:4 AY'CbCr)
pub const K_CV_PIXEL_FORMAT_TYPE_4444_AYPCBCR16: u32 = u32::from_be_bytes(*b"y416");

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CMTime {
    pub value: i64,
    pub timescale: i32,
    pub flags: u32,
    pub epoch: i64,
}

impl CMTime {
    pub const INVALID: CMTime = CMTime {
        value: 0,
        timescale: 0,
        flags: 0,
        epoch: 0,
    };

    #[inline]
    pub fn make(value: i64, timescale: i32) -> Self {
        Self {
            value,
            timescale,
            flags: 1, // kCMTimeFlags_Valid
            epoch: 0,
        }
    }
}

pub type VTCompressionOutputCallback = extern "C" fn(
    output_callback_ref_con: *mut c_void,
    source_frame_ref_con: *mut c_void,
    status: OSStatus,
    info_flags: u32,
    sample_buffer: CMSampleBufferRef,
);

#[link(name = "VideoToolbox", kind = "framework")]
extern "C" {
    // VideoToolbox session lifecycle & properties
    fn VTCompressionSessionCreate(
        allocator: CFAllocatorRef,
        width: i32,
        height: i32,
        codec_type: u32,
        encoder_specification: CFDictionaryRef,
        source_image_buffer_attributes: CFDictionaryRef,
        compressed_data_allocator: CFAllocatorRef,
        output_callback: VTCompressionOutputCallback,
        output_callback_ref_con: *mut c_void,
        compression_session_out: *mut VTCompressionSessionRef,
    ) -> OSStatus;

    fn VTSessionSetProperty(
        session: VTCompressionSessionRef,
        property_key: CFStringRef,
        property_value: CFTypeRef,
    ) -> OSStatus;

    fn VTCompressionSessionPrepareToEncodeFrames(session: VTCompressionSessionRef) -> OSStatus;

    fn VTCompressionSessionGetPixelBufferPool(
        session: VTCompressionSessionRef,
    ) -> CVPixelBufferPoolRef;

    fn VTCompressionSessionEncodeFrame(
        session: VTCompressionSessionRef,
        image_buffer: CVImageBufferRef,
        presentation_time_stamp: CMTime,
        duration: CMTime,
        frame_properties: CFDictionaryRef,
        source_frame_ref_con: *mut c_void,
        info_flags_out: *mut u32,
    ) -> OSStatus;

    fn VTCompressionSessionCompleteFrames(
        session: VTCompressionSessionRef,
        complete_until_presentation_time_stamp: CMTime,
    ) -> OSStatus;

    fn VTCompressionSessionInvalidate(session: VTCompressionSessionRef);

    // VideoToolbox property keys
    static kVTCompressionPropertyKey_AllowFrameReordering: CFStringRef;
    static kVTCompressionPropertyKey_ColorPrimaries: CFStringRef;
    static kVTCompressionPropertyKey_TransferFunction: CFStringRef;
    static kVTCompressionPropertyKey_YCbCrMatrix: CFStringRef;
}

#[link(name = "CoreMedia", kind = "framework")]
extern "C" {
    // CoreMedia color space constants
    static kCMFormatDescriptionColorPrimaries_ITU_R_2020: CFStringRef;
    static kCMFormatDescriptionTransferFunction_Linear: CFStringRef;
    static kCMFormatDescriptionYCbCrMatrix_ITU_R_2020: CFStringRef;

    // CoreMedia sample buffer extraction
    fn CMSampleBufferGetDataBuffer(sbuf: CMSampleBufferRef) -> CMBlockBufferRef;
    fn CMSampleBufferGetPresentationTimeStamp(sbuf: CMSampleBufferRef) -> CMTime;
    fn CMSampleBufferGetDuration(sbuf: CMSampleBufferRef) -> CMTime;
    fn CMBlockBufferGetDataLength(the_buffer: CMBlockBufferRef) -> usize;
    fn CMBlockBufferCopyDataBytes(
        the_source_buffer: CMBlockBufferRef,
        offset_to_data: usize,
        data_length: usize,
        destination: *mut c_void,
    ) -> OSStatus;
}

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    // CoreVideo pixel buffer pool & base address locking
    fn CVPixelBufferPoolCreatePixelBuffer(
        allocator: CFAllocatorRef,
        pixel_buffer_pool: CVPixelBufferPoolRef,
        pixel_buffer_out: *mut CVPixelBufferRef,
    ) -> CVReturn;

    fn CVPixelBufferCreate(
        allocator: CFAllocatorRef,
        width: usize,
        height: usize,
        pixel_format_type: u32,
        pixel_buffer_attributes: CFDictionaryRef,
        pixel_buffer_out: *mut CVPixelBufferRef,
    ) -> CVReturn;

    fn CVPixelBufferLockBaseAddress(pixel_buffer: CVPixelBufferRef, lock_flags: u64) -> CVReturn;
    fn CVPixelBufferUnlockBaseAddress(
        pixel_buffer: CVPixelBufferRef,
        unlock_flags: u64,
    ) -> CVReturn;
    fn CVPixelBufferGetBaseAddress(pixel_buffer: CVPixelBufferRef) -> *mut c_void;
    fn CVPixelBufferGetBytesPerRow(pixel_buffer: CVPixelBufferRef) -> usize;

    // CoreVideo image buffer color attachment keys
    fn CVBufferSetAttachment(
        buffer: CVPixelBufferRef,
        key: CFStringRef,
        value: CFTypeRef,
        attachment_mode: u32,
    );
    static kCVImageBufferColorPrimariesKey: CFStringRef;
    static kCVImageBufferTransferFunctionKey: CFStringRef;
    static kCVImageBufferYCbCrMatrixKey: CFStringRef;
}

/// Encoded ProRes frame packet emitted by `VTCompressionSession`.
#[derive(Debug, Clone)]
pub struct EncodedProResPacket {
    pub pts: u64,
    pub duration: u32,
    pub data: Vec<u8>,
}

struct CallbackState {
    packets: Mutex<Vec<Result<EncodedProResPacket, i32>>>,
}

extern "C" fn vt_compression_output_callback(
    output_callback_ref_con: *mut c_void,
    _source_frame_ref_con: *mut c_void,
    status: OSStatus,
    _info_flags: u32,
    sample_buffer: CMSampleBufferRef,
) {
    if output_callback_ref_con.is_null() {
        return;
    }
    let state = unsafe { &*(output_callback_ref_con as *const CallbackState) };

    if status != 0 || sample_buffer.is_null() {
        if let Ok(mut guard) = state.packets.lock() {
            guard.push(Err(status));
        }
        return;
    }

    unsafe {
        let block_buf = CMSampleBufferGetDataBuffer(sample_buffer);
        if block_buf.is_null() {
            return;
        }
        let data_len = CMBlockBufferGetDataLength(block_buf);
        let mut bytes = vec![0u8; data_len];
        let copy_status =
            CMBlockBufferCopyDataBytes(block_buf, 0, data_len, bytes.as_mut_ptr() as *mut c_void);
        if copy_status != 0 {
            if let Ok(mut guard) = state.packets.lock() {
                guard.push(Err(copy_status));
            }
            return;
        }

        let pts_cm = CMSampleBufferGetPresentationTimeStamp(sample_buffer);
        let dur_cm = CMSampleBufferGetDuration(sample_buffer);

        let packet = EncodedProResPacket {
            pts: pts_cm.value.max(0) as u64,
            duration: dur_cm.value.max(1) as u32,
            data: bytes,
        };

        if let Ok(mut guard) = state.packets.lock() {
            guard.push(Ok(packet));
        }
    }
}

/// Hardware Apple `VideoToolbox` ProRes 4444 / 422 HQ Encoder (`VTCompressionSession` + `CVPixelBufferPool`).
pub struct VideoToolboxProResEncoder {
    session: VTCompressionSessionRef,
    callback_state: Box<CallbackState>,
    pub width: u32,
    pub height: u32,
    pub timescale: u32,
    pub codec: ProResCodec,
}

unsafe impl Send for VideoToolboxProResEncoder {}

impl VideoToolboxProResEncoder {
    /// Creates and configures a hardware `VTCompressionSession` for Apple ProRes 4444 (`ap4h`)
    /// or Apple ProRes 422 HQ (`apch`) tagged with Rec. 2020 color primaries and matrix.
    pub fn new(
        width: u32,
        height: u32,
        timescale: u32,
        codec: ProResCodec,
    ) -> Result<Self, EncoderError> {
        let callback_state = Box::new(CallbackState {
            packets: Mutex::new(Vec::new()),
        });
        let state_ptr = (&*callback_state) as *const CallbackState as *mut c_void;

        let mut session: VTCompressionSessionRef = ptr::null_mut();
        let status = unsafe {
            VTCompressionSessionCreate(
                kCFAllocatorDefault,
                width as i32,
                height as i32,
                codec.cm_codec_type(),
                ptr::null(),
                ptr::null(),
                kCFAllocatorDefault,
                vt_compression_output_callback,
                state_ptr,
                &mut session,
            )
        };
        if status != 0 || session.is_null() {
            return Err(EncoderError::VideoToolbox(
                status,
                "VTCompressionSessionCreate",
            ));
        }

        unsafe {
            // ProRes is intra-frame only: disable frame reordering
            let _ = VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_AllowFrameReordering,
                kCFBooleanFalse as CFTypeRef,
            );
            // Set explicit ITU-R BT.2020 (`Rec. 2020`) color tags on the VideoToolbox session
            let _ = VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_ColorPrimaries,
                kCMFormatDescriptionColorPrimaries_ITU_R_2020 as CFTypeRef,
            );
            let _ = VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_TransferFunction,
                kCMFormatDescriptionTransferFunction_Linear as CFTypeRef,
            );
            let _ = VTSessionSetProperty(
                session,
                kVTCompressionPropertyKey_YCbCrMatrix,
                kCMFormatDescriptionYCbCrMatrix_ITU_R_2020 as CFTypeRef,
            );

            let prep_status = VTCompressionSessionPrepareToEncodeFrames(session);
            if prep_status != 0 {
                VTCompressionSessionInvalidate(session);
                CFRelease(session as CFTypeRef);
                return Err(EncoderError::VideoToolbox(
                    prep_status,
                    "VTCompressionSessionPrepareToEncodeFrames",
                ));
            }
        }

        Ok(Self {
            session,
            callback_state,
            width,
            height,
            timescale,
            codec,
        })
    }

    /// Allocates a `CVPixelBuffer` (`kCVPixelFormatType_64RGBAHalf`), runs the Metal compute
    /// shader directly into the locked `CVPixelBuffer` unified memory, attaches Rec. 2020
    /// `nclc` color tags, and submits the frame to `VTCompressionSessionEncodeFrame`.
    pub fn encode_bayer_frame_with_metal(
        &mut self,
        gpu: &mut MetalColorPipeline,
        bayer_rggb: &[u16],
        params: &ColorPipelineParams,
        pts: u64,
        duration: u32,
    ) -> Result<Vec<EncodedProResPacket>, EncoderError> {
        let mut pixel_buffer: CVPixelBufferRef = ptr::null_mut();

        // Allocate `kCVPixelFormatType_64RGBAHalf` (`RGhA`) pixel buffer for 12-bit ProRes precision
        let cv_status = unsafe {
            CVPixelBufferCreate(
                kCFAllocatorDefault,
                self.width as usize,
                self.height as usize,
                K_CV_PIXEL_FORMAT_TYPE_64_RGBA_HALF,
                ptr::null(),
                &mut pixel_buffer,
            )
        };
        if cv_status != 0 || pixel_buffer.is_null() {
            // Fallback to session's CVPixelBufferPool if direct creation failed
            let pool = unsafe { VTCompressionSessionGetPixelBufferPool(self.session) };
            if !pool.is_null() {
                let pool_status = unsafe {
                    CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &mut pixel_buffer)
                };
                if pool_status != 0 || pixel_buffer.is_null() {
                    return Err(EncoderError::CoreVideo(pool_status));
                }
            } else {
                return Err(EncoderError::CoreVideo(cv_status));
            }
        }

        // Lock CVPixelBuffer base address and execute Metal compute shader directly into it
        let lock_res = unsafe { CVPixelBufferLockBaseAddress(pixel_buffer, 0) };
        if lock_res != 0 {
            unsafe { CFRelease(pixel_buffer as CFTypeRef) };
            return Err(EncoderError::CoreVideo(lock_res));
        }

        let base_addr = unsafe { CVPixelBufferGetBaseAddress(pixel_buffer) };
        let bytes_per_row = unsafe { CVPixelBufferGetBytesPerRow(pixel_buffer) };
        let total_bytes = (self.height as usize) * bytes_per_row;

        let gpu_res = gpu.process_into_pixel_buffer_memory(
            bayer_rggb,
            params,
            base_addr,
            bytes_per_row,
            total_bytes,
            false,
        );

        unsafe {
            CVPixelBufferUnlockBaseAddress(pixel_buffer, 0);
        }

        if let Err(e) = gpu_res {
            unsafe { CFRelease(pixel_buffer as CFTypeRef) };
            return Err(EncoderError::Metal(e));
        }

        // Attach ITU-R BT.2020 (`nclc`) color tags to the CVPixelBuffer (`kCVAttachmentMode_ShouldPropagate = 1`)
        unsafe {
            CVBufferSetAttachment(
                pixel_buffer,
                kCVImageBufferColorPrimariesKey,
                kCMFormatDescriptionColorPrimaries_ITU_R_2020 as CFTypeRef,
                1,
            );
            CVBufferSetAttachment(
                pixel_buffer,
                kCVImageBufferTransferFunctionKey,
                kCMFormatDescriptionTransferFunction_Linear as CFTypeRef,
                1,
            );
            CVBufferSetAttachment(
                pixel_buffer,
                kCVImageBufferYCbCrMatrixKey,
                kCMFormatDescriptionYCbCrMatrix_ITU_R_2020 as CFTypeRef,
                1,
            );
        }

        let cm_pts = CMTime::make(pts as i64, self.timescale as i32);
        let cm_dur = CMTime::make(duration as i64, self.timescale as i32);
        let mut info_flags = 0u32;

        let enc_status = unsafe {
            VTCompressionSessionEncodeFrame(
                self.session,
                pixel_buffer,
                cm_pts,
                cm_dur,
                ptr::null(),
                ptr::null_mut(),
                &mut info_flags,
            )
        };

        unsafe {
            CFRelease(pixel_buffer as CFTypeRef);
        }

        if enc_status != 0 {
            return Err(EncoderError::VideoToolbox(
                enc_status,
                "VTCompressionSessionEncodeFrame",
            ));
        }

        self.drain_emitted_packets()
    }

    /// Flushes all pending frames in the `VTCompressionSession` and returns any remaining packets.
    pub fn finish(&mut self) -> Result<Vec<EncodedProResPacket>, EncoderError> {
        let status = unsafe { VTCompressionSessionCompleteFrames(self.session, CMTime::INVALID) };
        if status != 0 {
            return Err(EncoderError::VideoToolbox(
                status,
                "VTCompressionSessionCompleteFrames",
            ));
        }
        self.drain_emitted_packets()
    }

    fn drain_emitted_packets(&self) -> Result<Vec<EncodedProResPacket>, EncoderError> {
        let mut guard = self.callback_state.packets.lock().unwrap();
        let mut out = Vec::with_capacity(guard.len());
        for res in guard.drain(..) {
            match res {
                Ok(pkt) => out.push(pkt),
                Err(code) => {
                    return Err(EncoderError::VideoToolbox(
                        code,
                        "VTCompressionOutputCallback",
                    ))
                }
            }
        }
        Ok(out)
    }
}

impl Drop for VideoToolboxProResEncoder {
    fn drop(&mut self) {
        if !self.session.is_null() {
            unsafe {
                VTCompressionSessionInvalidate(self.session);
                CFRelease(self.session as CFTypeRef);
            }
            self.session = ptr::null_mut();
        }
    }
}

// ============================================================================
// Native 64-Bit QuickTime (`.mov`) Container Writer with Rec.2020 `colr` (`nclc`),
// 24-Bit Linear PCM Audio Passthrough, and Camera Timecode (`tmcd`) Passthrough
// ============================================================================

#[derive(Debug, Clone)]
struct WrittenVideoSample {
    file_offset: u64,
    byte_size: u32,
    duration: u32,
}

#[derive(Debug, Clone)]
struct WrittenAudioChunk {
    file_offset: u64,
    sample_count: u32,
}

/// Streaming 64-bit QuickTime `.mov` container muxer.
///
/// Writes `ftyp` (`qt  `), streams encoded ProRes frames + PCM audio + timecode into a
/// 64-bit extended `mdat` atom (`size == 1`), and finalizes the `moov` atom with:
/// - Explicit `colr` (`nclc`) atom:
///   - Color Primaries = `9` (`kCMFormatDescriptionColorPrimaries_ITU_R_2020`)
///   - Transfer Function = `8` (`kCMFormatDescriptionTransferFunction_Linear` / N-Log metadata)
///   - Matrix = `9` (`kCMFormatDescriptionYCbCrMatrix_ITU_R_2020`)
/// - Uncompressed 24-bit Linear PCM audio track (`in24`)
/// - Camera Timecode track (`tmcd`) linked via `tref -> tmcd`
pub struct QuickTimeMovWriter {
    writer: BufWriter<File>,
    width: u16,
    height: u16,
    timescale: u32,
    codec: ProResCodec,
    mdat_start_offset: u64,
    video_samples: Vec<WrittenVideoSample>,
    audio_chunks: Vec<WrittenAudioChunk>,
    tmcd_sample_offset: Option<u64>,
}

impl QuickTimeMovWriter {
    pub fn create<P: AsRef<Path>>(
        output_path: P,
        width: u32,
        height: u32,
        timescale: u32,
        codec: ProResCodec,
    ) -> Result<Self, EncoderError> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::with_capacity(4 * 1024 * 1024, file);

        // 1. Write QuickTime `ftyp` atom (`qt  `)
        let mut ftyp = Vec::with_capacity(24);
        ftyp.extend_from_slice(&20u32.to_be_bytes());
        ftyp.extend_from_slice(b"ftyp");
        ftyp.extend_from_slice(b"qt  ");
        ftyp.extend_from_slice(&0x0000_0200u32.to_be_bytes());
        ftyp.extend_from_slice(b"qt  ");
        writer.write_all(&ftyp)?;

        // 2. Begin 64-bit extended `mdat` atom (`size == 1` + `'mdat'` + 8-byte `largesize`)
        let mdat_start_offset = writer.stream_position()?;
        writer.write_all(&1u32.to_be_bytes())?;
        writer.write_all(b"mdat")?;
        writer.write_all(&0u64.to_be_bytes())?; // Patched in `finalize()`

        Ok(Self {
            writer,
            width: width as u16,
            height: height as u16,
            timescale: timescale.max(1),
            codec,
            mdat_start_offset,
            video_samples: Vec::new(),
            audio_chunks: Vec::new(),
            tmcd_sample_offset: None,
        })
    }

    /// Writes an encoded ProRes video packet into the `mdat` stream.
    pub fn write_video_packet(&mut self, packet: &EncodedProResPacket) -> Result<(), EncoderError> {
        let file_offset = self.writer.stream_position()?;
        self.writer.write_all(&packet.data)?;
        self.video_samples.push(WrittenVideoSample {
            file_offset,
            byte_size: packet.data.len() as u32,
            duration: packet.duration,
        });
        Ok(())
    }

    /// Copies the uncompressed 24-bit Linear PCM audio chunks from the source `.nev` into `mdat`.
    pub fn passthrough_audio_track<R: Read + Seek>(
        &mut self,
        nev_reader: &mut R,
        audio: &AudioTrackInfo,
    ) -> Result<(), EncoderError> {
        let mut buf = Vec::new();
        for chunk in &audio.chunks {
            nev_reader.seek(SeekFrom::Start(chunk.file_offset))?;
            buf.resize(chunk.byte_size as usize, 0);
            nev_reader.read_exact(&mut buf)?;

            let dst_offset = self.writer.stream_position()?;
            self.writer.write_all(&buf)?;
            self.audio_chunks.push(WrittenAudioChunk {
                file_offset: dst_offset,
                sample_count: chunk.sample_count,
            });
        }
        Ok(())
    }

    /// Writes the original camera timecode (`tmcd`) sample into `mdat`.
    pub fn passthrough_timecode_sample(
        &mut self,
        tc: &TimecodeTrackInfo,
    ) -> Result<(), EncoderError> {
        let offset = self.writer.stream_position()?;
        self.writer.write_all(&tc.frame_number.to_be_bytes())?;
        self.tmcd_sample_offset = Some(offset);
        Ok(())
    }

    /// Patches the 64-bit `mdat` size and writes the complete `moov` atom at the end of the `.mov` file.
    pub fn finalize(
        mut self,
        audio_info: Option<&AudioTrackInfo>,
        tmcd_info: Option<&TimecodeTrackInfo>,
    ) -> Result<(), EncoderError> {
        let mdat_end_offset = self.writer.stream_position()?;
        let mdat_size = mdat_end_offset - self.mdat_start_offset;

        // Patch 64-bit extended size in `mdat` header
        self.writer
            .seek(SeekFrom::Start(self.mdat_start_offset + 8))?;
        self.writer.write_all(&mdat_size.to_be_bytes())?;
        self.writer.seek(SeekFrom::Start(mdat_end_offset))?;

        let total_video_duration: u64 = self.video_samples.iter().map(|s| s.duration as u64).sum();

        let has_tmcd = tmcd_info.is_some() && self.tmcd_sample_offset.is_some();
        let next_track_id = 4u32;

        let mut moov = Vec::new();

        // 1. `mvhd` (Movie Header Atom)
        let mvhd = build_mvhd_atom(self.timescale, total_video_duration, next_track_id);
        push_atom(&mut moov, b"mvhd", &mvhd);

        // 2. Video `trak` (Track ID 1: ProRes 4444 `ap4h` or ProRes 422 HQ `apch` with `colr` `nclc`)
        let video_trak = build_video_trak_atom(
            1,
            self.width,
            self.height,
            self.timescale,
            total_video_duration,
            self.codec,
            &self.video_samples,
            if has_tmcd { Some(3) } else { None },
        );
        push_atom(&mut moov, b"trak", &video_trak);

        // 3. Optional Audio `trak` (Track ID 2: 24-bit Linear PCM passthrough)
        if let Some(audio) = audio_info {
            if !self.audio_chunks.is_empty() {
                let audio_trak = build_audio_trak_atom(2, audio, &self.audio_chunks);
                push_atom(&mut moov, b"trak", &audio_trak);
            }
        }

        // 4. Optional Timecode `trak` (Track ID 3: `tmcd` passthrough)
        if let (Some(tc), Some(tc_offset)) = (tmcd_info, self.tmcd_sample_offset) {
            let tmcd_trak = build_tmcd_trak_atom(3, tc, total_video_duration, tc_offset);
            push_atom(&mut moov, b"trak", &tmcd_trak);
        }

        // 5. `udta` atom with Nikon N-Log / Rec. 2020 color & camera metadata
        let mut udta = Vec::new();
        push_atom(
            &mut udta,
            b"NLOG",
            b"Nikon N-Log / ITU-R BT.2020 Scene-Linear RAW Transcode",
        );
        push_atom(&mut moov, b"udta", &udta);

        let mut moov_box = Vec::with_capacity(8 + moov.len());
        push_atom(&mut moov_box, b"moov", &moov);

        self.writer.write_all(&moov_box)?;
        self.writer.flush()?;
        Ok(())
    }
}

fn push_atom(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    let size = (8 + payload.len()) as u32;
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(payload);
}

fn build_mvhd_atom(timescale: u32, duration: u64, next_track_id: u32) -> Vec<u8> {
    let mut mvhd = vec![0u8; 100];
    mvhd[0] = 0; // version 0
    mvhd[12..16].copy_from_slice(&timescale.to_be_bytes());
    mvhd[16..20].copy_from_slice(&(duration as u32).to_be_bytes());
    mvhd[20..24].copy_from_slice(&0x0001_0000u32.to_be_bytes()); // preferred rate 1.0
    mvhd[24..26].copy_from_slice(&0x0100u16.to_be_bytes()); // preferred volume 1.0
                                                            // Unity matrix (3x3 fixed point)
    mvhd[36..40].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    mvhd[52..56].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    mvhd[68..72].copy_from_slice(&0x4000_0000u32.to_be_bytes());
    mvhd[96..100].copy_from_slice(&next_track_id.to_be_bytes());
    mvhd
}

#[allow(clippy::too_many_arguments)]
fn build_video_trak_atom(
    track_id: u32,
    width: u16,
    height: u16,
    timescale: u32,
    duration: u64,
    codec: ProResCodec,
    samples: &[WrittenVideoSample],
    tmcd_track_id: Option<u32>,
) -> Vec<u8> {
    let mut trak = Vec::new();

    // `tkhd` (Track Header)
    let mut tkhd = vec![0u8; 84];
    tkhd[3] = 0x0F; // Track enabled, in movie, in preview, in poster
    tkhd[12..16].copy_from_slice(&track_id.to_be_bytes());
    tkhd[20..24].copy_from_slice(&(duration as u32).to_be_bytes());
    tkhd[40..44].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    tkhd[56..60].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    tkhd[72..76].copy_from_slice(&0x4000_0000u32.to_be_bytes());
    tkhd[76..80].copy_from_slice(&((width as u32) << 16).to_be_bytes());
    tkhd[80..84].copy_from_slice(&((height as u32) << 16).to_be_bytes());
    push_atom(&mut trak, b"tkhd", &tkhd);

    // Optional `tref -> tmcd` linking video track to timecode track
    if let Some(tc_id) = tmcd_track_id {
        let mut tref = Vec::new();
        push_atom(&mut tref, b"tmcd", &tc_id.to_be_bytes());
        push_atom(&mut trak, b"tref", &tref);
    }

    // `mdia`
    let mut mdia = Vec::new();
    let mut mdhd = vec![0u8; 24];
    mdhd[12..16].copy_from_slice(&timescale.to_be_bytes());
    mdhd[16..20].copy_from_slice(&(duration as u32).to_be_bytes());
    push_atom(&mut mdia, b"mdhd", &mdhd);

    let mut hdlr = vec![0u8; 24];
    hdlr[4..8].copy_from_slice(b"mhlr");
    hdlr[8..12].copy_from_slice(b"vide");
    hdlr[12..16].copy_from_slice(b"appl");
    push_atom(&mut mdia, b"hdlr", &hdlr);

    // `minf`
    let mut minf = Vec::new();
    let mut vmhd = vec![0u8; 12];
    vmhd[3] = 1;
    push_atom(&mut minf, b"vmhd", &vmhd);

    let dinf = build_dinf_self_contained();
    push_atom(&mut minf, b"dinf", &dinf);

    // `stbl`
    let mut stbl = Vec::new();

    // `stsd` with ProRes (`ap4h` or `apch`) and explicit `colr` (`nclc`) atom
    let mut stsd = Vec::new();
    stsd.extend_from_slice(&0u32.to_be_bytes());
    stsd.extend_from_slice(&1u32.to_be_bytes()); // 1 entry

    let mut sample_entry = vec![0u8; 86];
    sample_entry[4..8].copy_from_slice(&codec.fourcc());
    sample_entry[14..16].copy_from_slice(&1u16.to_be_bytes()); // data_reference_index = 1
    sample_entry[20..24].copy_from_slice(b"appl"); // vendor
    sample_entry[24..28].copy_from_slice(&0x0000_03FFu32.to_be_bytes()); // temporal quality
    sample_entry[28..32].copy_from_slice(&0x0000_03FFu32.to_be_bytes()); // spatial quality
    sample_entry[32..34].copy_from_slice(&width.to_be_bytes());
    sample_entry[34..36].copy_from_slice(&height.to_be_bytes());
    sample_entry[36..40].copy_from_slice(&0x0048_0000u32.to_be_bytes()); // 72 dpi horiz
    sample_entry[40..44].copy_from_slice(&0x0048_0000u32.to_be_bytes()); // 72 dpi vert
    sample_entry[48..50].copy_from_slice(&1u16.to_be_bytes()); // frame_count = 1
    let compressor_name: &[u8] = match codec {
        ProResCodec::ProRes4444 => b"\x11Apple ProRes 4444",
        ProResCodec::ProRes422HQ => b"\x13Apple ProRes 422 HQ",
    };
    sample_entry[50..50 + compressor_name.len()].copy_from_slice(compressor_name);
    let depth: u16 = match codec {
        ProResCodec::ProRes4444 => 32, // 4:4:4:4 container depth
        ProResCodec::ProRes422HQ => 24,
    };
    sample_entry[82..84].copy_from_slice(&depth.to_be_bytes());
    sample_entry[84..86].copy_from_slice(&(-1i16).to_be_bytes()); // color_table_id = -1

    // Append `colr` (`nclc`) extension atom inside the VideoSampleEntry:
    //   Color Parameter Type: 'nclc'
    //   Primaries Index: 9 (ITU-R BT.2020 / kCMFormatDescriptionColorPrimaries_ITU_R_2020)
    //   Transfer Function Index: 8 (Linear / N-Log Wide Gamut HDR tag)
    //   Matrix Index: 9 (ITU-R BT.2020 non-constant luminance / kCMFormatDescriptionYCbCrMatrix_ITU_R_2020)
    let mut colr_payload = Vec::with_capacity(10);
    colr_payload.extend_from_slice(b"nclc");
    colr_payload.extend_from_slice(&9u16.to_be_bytes()); // ITU_R_2020 Primaries
    colr_payload.extend_from_slice(&8u16.to_be_bytes()); // Linear / N-Log Transfer Function
    colr_payload.extend_from_slice(&9u16.to_be_bytes()); // ITU_R_2020 Matrix
    push_atom(&mut sample_entry, b"colr", &colr_payload);

    // Append `fiel` (progressive scan = 1, 0) and `pasp` (1:1 aspect ratio)
    push_atom(&mut sample_entry, b"fiel", &[1u8, 0u8]);
    push_atom(&mut sample_entry, b"pasp", &[0, 0, 0, 1, 0, 0, 0, 1]);

    let entry_total_len = sample_entry.len() as u32;
    sample_entry[0..4].copy_from_slice(&entry_total_len.to_be_bytes());
    stsd.extend_from_slice(&sample_entry);
    push_atom(&mut stbl, b"stsd", &stsd);

    // `stts` (run-length encoded frame durations for both CFR and VFR clips)
    let mut rle_stts: Vec<(u32, u32)> = Vec::new();
    for s in samples {
        if let Some(last) = rle_stts.last_mut() {
            if last.1 == s.duration {
                last.0 += 1;
                continue;
            }
        }
        rle_stts.push((1, s.duration));
    }
    let mut stts = Vec::new();
    stts.extend_from_slice(&0u32.to_be_bytes());
    stts.extend_from_slice(&(rle_stts.len() as u32).to_be_bytes());
    for &(count, delta) in &rle_stts {
        stts.extend_from_slice(&count.to_be_bytes());
        stts.extend_from_slice(&delta.to_be_bytes());
    }
    push_atom(&mut stbl, b"stts", &stts);

    // `stsc` (1 sample per chunk)
    let mut stsc = Vec::new();
    stsc.extend_from_slice(&0u32.to_be_bytes());
    stsc.extend_from_slice(&1u32.to_be_bytes());
    stsc.extend_from_slice(&1u32.to_be_bytes());
    stsc.extend_from_slice(&1u32.to_be_bytes());
    stsc.extend_from_slice(&1u32.to_be_bytes());
    push_atom(&mut stbl, b"stsc", &stsc);

    // `stsz`
    let mut stsz = Vec::new();
    stsz.extend_from_slice(&0u32.to_be_bytes());
    stsz.extend_from_slice(&0u32.to_be_bytes()); // variable per-frame sizes
    stsz.extend_from_slice(&(samples.len() as u32).to_be_bytes());
    for s in samples {
        stsz.extend_from_slice(&s.byte_size.to_be_bytes());
    }
    push_atom(&mut stbl, b"stsz", &stsz);

    // `co64` (64-bit chunk offsets)
    let mut co64 = Vec::new();
    co64.extend_from_slice(&0u32.to_be_bytes());
    co64.extend_from_slice(&(samples.len() as u32).to_be_bytes());
    for s in samples {
        co64.extend_from_slice(&s.file_offset.to_be_bytes());
    }
    push_atom(&mut stbl, b"co64", &co64);

    push_atom(&mut minf, b"stbl", &stbl);
    push_atom(&mut mdia, b"minf", &minf);
    push_atom(&mut trak, b"mdia", &mdia);
    trak
}

fn build_audio_trak_atom(
    track_id: u32,
    audio: &AudioTrackInfo,
    chunks: &[WrittenAudioChunk],
) -> Vec<u8> {
    let mut trak = Vec::new();
    let total_samples: u64 = chunks.iter().map(|c| c.sample_count as u64).sum();

    let mut tkhd = vec![0u8; 84];
    tkhd[3] = 0x07;
    tkhd[12..16].copy_from_slice(&track_id.to_be_bytes());
    tkhd[20..24].copy_from_slice(&(total_samples as u32).to_be_bytes());
    tkhd[36..38].copy_from_slice(&0x0100u16.to_be_bytes()); // volume 1.0
    tkhd[40..44].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    tkhd[56..60].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    tkhd[72..76].copy_from_slice(&0x4000_0000u32.to_be_bytes());
    push_atom(&mut trak, b"tkhd", &tkhd);

    let mut mdia = Vec::new();
    let mut mdhd = vec![0u8; 24];
    mdhd[12..16].copy_from_slice(&audio.sample_rate.to_be_bytes());
    mdhd[16..20].copy_from_slice(&(total_samples as u32).to_be_bytes());
    push_atom(&mut mdia, b"mdhd", &mdhd);

    let mut hdlr = vec![0u8; 24];
    hdlr[4..8].copy_from_slice(b"mhlr");
    hdlr[8..12].copy_from_slice(b"soun");
    hdlr[12..16].copy_from_slice(b"appl");
    push_atom(&mut mdia, b"hdlr", &hdlr);

    let mut minf = Vec::new();
    let smhd = vec![0u8; 8];
    push_atom(&mut minf, b"smhd", &smhd);
    let dinf = build_dinf_self_contained();
    push_atom(&mut minf, b"dinf", &dinf);

    let mut stbl = Vec::new();
    let mut stsd = Vec::new();
    stsd.extend_from_slice(&0u32.to_be_bytes());
    stsd.extend_from_slice(&1u32.to_be_bytes());
    let mut entry = vec![0u8; 36];
    entry[0..4].copy_from_slice(&36u32.to_be_bytes());
    entry[4..8].copy_from_slice(&audio.codec_fourcc);
    entry[14..16].copy_from_slice(&1u16.to_be_bytes());
    entry[24..26].copy_from_slice(&audio.channels.to_be_bytes());
    entry[26..28].copy_from_slice(&audio.sample_bits.to_be_bytes());
    entry[32..36].copy_from_slice(&(audio.sample_rate << 16).to_be_bytes());
    stsd.extend_from_slice(&entry);
    push_atom(&mut stbl, b"stsd", &stsd);

    let mut stts = Vec::new();
    stts.extend_from_slice(&0u32.to_be_bytes());
    stts.extend_from_slice(&1u32.to_be_bytes());
    stts.extend_from_slice(&(total_samples as u32).to_be_bytes());
    stts.extend_from_slice(&1u32.to_be_bytes());
    push_atom(&mut stbl, b"stts", &stts);

    let mut stsc = Vec::new();
    stsc.extend_from_slice(&0u32.to_be_bytes());
    stsc.extend_from_slice(&(chunks.len() as u32).to_be_bytes());
    for (idx, c) in chunks.iter().enumerate() {
        stsc.extend_from_slice(&((idx + 1) as u32).to_be_bytes());
        stsc.extend_from_slice(&c.sample_count.to_be_bytes());
        stsc.extend_from_slice(&1u32.to_be_bytes());
    }
    push_atom(&mut stbl, b"stsc", &stsc);

    let bytes_per_frame = (audio.channels as u32) * ((audio.sample_bits as u32) / 8).max(1);
    let mut stsz = Vec::new();
    stsz.extend_from_slice(&0u32.to_be_bytes());
    stsz.extend_from_slice(&bytes_per_frame.to_be_bytes());
    stsz.extend_from_slice(&(total_samples as u32).to_be_bytes());
    push_atom(&mut stbl, b"stsz", &stsz);

    let mut co64 = Vec::new();
    co64.extend_from_slice(&0u32.to_be_bytes());
    co64.extend_from_slice(&(chunks.len() as u32).to_be_bytes());
    for c in chunks {
        co64.extend_from_slice(&c.file_offset.to_be_bytes());
    }
    push_atom(&mut stbl, b"co64", &co64);

    push_atom(&mut minf, b"stbl", &stbl);
    push_atom(&mut mdia, b"minf", &minf);
    push_atom(&mut trak, b"mdia", &mdia);
    trak
}

fn build_tmcd_trak_atom(
    track_id: u32,
    tc: &TimecodeTrackInfo,
    total_duration: u64,
    sample_offset: u64,
) -> Vec<u8> {
    let mut trak = Vec::new();

    let mut tkhd = vec![0u8; 84];
    tkhd[3] = 0x07;
    tkhd[12..16].copy_from_slice(&track_id.to_be_bytes());
    tkhd[20..24].copy_from_slice(&(total_duration as u32).to_be_bytes());
    tkhd[40..44].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    tkhd[56..60].copy_from_slice(&0x0001_0000u32.to_be_bytes());
    tkhd[72..76].copy_from_slice(&0x4000_0000u32.to_be_bytes());
    push_atom(&mut trak, b"tkhd", &tkhd);

    let mut mdia = Vec::new();
    let mut mdhd = vec![0u8; 24];
    mdhd[12..16].copy_from_slice(&tc.timescale.to_be_bytes());
    mdhd[16..20].copy_from_slice(&(total_duration as u32).to_be_bytes());
    push_atom(&mut mdia, b"mdhd", &mdhd);

    let mut hdlr = vec![0u8; 24];
    hdlr[4..8].copy_from_slice(b"mhlr");
    hdlr[8..12].copy_from_slice(b"tmcd");
    hdlr[12..16].copy_from_slice(b"appl");
    push_atom(&mut mdia, b"hdlr", &hdlr);

    let mut minf = Vec::new();
    let gmhd = vec![0u8; 16];
    push_atom(&mut minf, b"gmhd", &gmhd);
    let dinf = build_dinf_self_contained();
    push_atom(&mut minf, b"dinf", &dinf);

    let mut stbl = Vec::new();
    let mut stsd = Vec::new();
    stsd.extend_from_slice(&0u32.to_be_bytes());
    stsd.extend_from_slice(&1u32.to_be_bytes());
    let mut entry = vec![0u8; 38];
    entry[0..4].copy_from_slice(&38u32.to_be_bytes());
    entry[4..8].copy_from_slice(b"tmcd");
    entry[14..16].copy_from_slice(&1u16.to_be_bytes());
    entry[20..24].copy_from_slice(&tc.flags.to_be_bytes());
    entry[24..28].copy_from_slice(&tc.timescale.to_be_bytes());
    entry[28..32].copy_from_slice(&tc.frame_duration.to_be_bytes());
    entry[32] = tc.num_frames;
    stsd.extend_from_slice(&entry);
    push_atom(&mut stbl, b"stsd", &stsd);

    let mut stts = Vec::new();
    stts.extend_from_slice(&0u32.to_be_bytes());
    stts.extend_from_slice(&1u32.to_be_bytes());
    stts.extend_from_slice(&1u32.to_be_bytes());
    stts.extend_from_slice(&(total_duration as u32).to_be_bytes());
    push_atom(&mut stbl, b"stts", &stts);

    let mut stsc = Vec::new();
    stsc.extend_from_slice(&0u32.to_be_bytes());
    stsc.extend_from_slice(&1u32.to_be_bytes());
    stsc.extend_from_slice(&1u32.to_be_bytes());
    stsc.extend_from_slice(&1u32.to_be_bytes());
    stsc.extend_from_slice(&1u32.to_be_bytes());
    push_atom(&mut stbl, b"stsc", &stsc);

    let mut stsz = Vec::new();
    stsz.extend_from_slice(&0u32.to_be_bytes());
    stsz.extend_from_slice(&4u32.to_be_bytes());
    stsz.extend_from_slice(&1u32.to_be_bytes());
    push_atom(&mut stbl, b"stsz", &stsz);

    let mut co64 = Vec::new();
    co64.extend_from_slice(&0u32.to_be_bytes());
    co64.extend_from_slice(&1u32.to_be_bytes());
    co64.extend_from_slice(&sample_offset.to_be_bytes());
    push_atom(&mut stbl, b"co64", &co64);

    push_atom(&mut minf, b"stbl", &stbl);
    push_atom(&mut mdia, b"minf", &minf);
    push_atom(&mut trak, b"mdia", &mdia);
    trak
}

fn build_dinf_self_contained() -> Vec<u8> {
    let mut dref = Vec::new();
    dref.extend_from_slice(&0u32.to_be_bytes());
    dref.extend_from_slice(&1u32.to_be_bytes()); // 1 entry
    push_atom(&mut dref, b"alis", &[0, 0, 0, 1]); // self-contained flag = 0x000001
    let mut dinf = Vec::new();
    push_atom(&mut dinf, b"dref", &dref);
    dinf
}
