use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

use crate::metadata::{parse_ncdt_payload, NikonMetadata};

#[derive(Debug, Error)]
pub enum DemuxError {
    #[error("I/O error while reading .nev container: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid or unsupported ISO-BMFF/QuickTime container: {0}")]
    InvalidContainer(String),
    #[error("No video track found in .nev container")]
    MissingVideoTrack,
    #[error("Incomplete sample table (stbl) in video track: {0}")]
    IncompleteSampleTable(&'static str),
}

/// Exact resolved byte location and presentation timing for a single video frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameDescriptor {
    pub frame_index: u32,
    pub file_offset: u64,
    pub byte_size: u32,
    pub pts: u64,
    pub duration: u32,
}

/// Byte location for an uncompressed Linear PCM audio chunk in the source `.nev`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioChunkDescriptor {
    pub file_offset: u64,
    pub byte_size: u32,
    pub sample_count: u32,
}

/// Extracted Linear PCM Audio Track configuration and chunk map.
#[derive(Debug, Clone)]
pub struct AudioTrackInfo {
    pub codec_fourcc: [u8; 4],
    pub channels: u16,
    pub sample_bits: u16,
    pub sample_rate: u32,
    pub timescale: u32,
    pub total_samples: u64,
    pub chunks: Vec<AudioChunkDescriptor>,
}

/// Extracted Camera Timecode (`tmcd`) Track configuration.
#[derive(Debug, Clone)]
pub struct TimecodeTrackInfo {
    pub flags: u32,
    pub timescale: u32,
    pub frame_duration: u32,
    pub num_frames: u8,
    pub frame_number: u32,
    pub timecode_str: String,
}

/// Complete demuxed information for a Nikon `.nev` container.
#[derive(Debug, Clone)]
pub struct NevDemuxedContainer {
    pub major_brand: [u8; 4],
    pub video_codec_fourcc: [u8; 4],
    pub width: u32,
    pub height: u32,
    pub bit_depth: u16,
    pub timescale: u32,
    pub duration: u64,
    pub frames: Vec<FrameDescriptor>,
    pub audio_track: Option<AudioTrackInfo>,
    pub timecode_track: Option<TimecodeTrackInfo>,
    pub metadata: NikonMetadata,
}

impl NevDemuxedContainer {
    /// Computes the nominal frame rate as `(fps_num, fps_den)` and floating-point FPS.
    pub fn frame_rate(&self) -> (u32, u32, f64) {
        if let Some(first) = self.frames.first() {
            if first.duration > 0 && self.timescale > 0 {
                let fps = self.timescale as f64 / first.duration as f64;
                return (self.timescale, first.duration, fps);
            }
        }
        (24000, 1001, 23.976)
    }

    /// Reads the compressed TicoRAW bitstream bytes for `frame` directly from `reader`.
    pub fn read_frame_bytes<R: Read + Seek>(
        &self,
        reader: &mut R,
        frame: &FrameDescriptor,
    ) -> Result<Vec<u8>, DemuxError> {
        reader.seek(SeekFrom::Start(frame.file_offset))?;
        let mut buf = vec![0u8; frame.byte_size as usize];
        reader.read_exact(&mut buf)?;
        Ok(buf)
    }
}

/// Parsed 32-bit or 64-bit ISO-BMFF Box Header.
#[derive(Debug, Clone, Copy)]
pub struct BoxHeader {
    pub fourcc: [u8; 4],
    pub start_offset: u64,
    pub payload_offset: u64,
    pub total_size: u64,
}

impl BoxHeader {
    #[inline]
    pub fn payload_size(&self) -> u64 {
        self.total_size
            .saturating_sub(self.payload_offset - self.start_offset)
    }

    #[inline]
    pub fn end_offset(&self) -> u64 {
        self.start_offset.saturating_add(self.total_size)
    }

    pub fn fourcc_str(&self) -> String {
        String::from_utf8_lossy(&self.fourcc).to_string()
    }
}

/// Reads the next ISO-BMFF box header at the current cursor position,
/// supporting 64-bit extended box sizes (`size == 1` followed by 64-bit `largesize`).
pub fn read_box_header<R: Read + Seek>(
    reader: &mut R,
    parent_end: u64,
) -> Result<Option<BoxHeader>, DemuxError> {
    let start_offset = reader.stream_position()?;
    if start_offset + 8 > parent_end {
        return Ok(None);
    }

    let mut hdr = [0u8; 8];
    reader.read_exact(&mut hdr)?;
    let size32 = u32::from_be_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]);
    let fourcc = [hdr[4], hdr[5], hdr[6], hdr[7]];

    let (total_size, header_len) = if size32 == 1 {
        if start_offset + 16 > parent_end {
            return Err(DemuxError::InvalidContainer(
                "Truncated 64-bit extended box header".into(),
            ));
        }
        let mut ext = [0u8; 8];
        reader.read_exact(&mut ext)?;
        let size64 = u64::from_be_bytes(ext);
        if size64 < 16 {
            return Err(DemuxError::InvalidContainer(format!(
                "Invalid 64-bit box size {} for '{}'",
                size64,
                String::from_utf8_lossy(&fourcc)
            )));
        }
        (size64, 16u64)
    } else if size32 == 0 {
        // Box extends to the end of the enclosing container/file
        (parent_end.saturating_sub(start_offset), 8u64)
    } else {
        if (size32 as u64) < 8 {
            return Err(DemuxError::InvalidContainer(format!(
                "Invalid 32-bit box size {} for '{}'",
                size32,
                String::from_utf8_lossy(&fourcc)
            )));
        }
        (size32 as u64, 8u64)
    };

    let payload_offset = start_offset + header_len;
    if start_offset + total_size > parent_end {
        return Err(DemuxError::InvalidContainer(format!(
            "Box '{}' at offset {} exceeds parent boundary ({} > {})",
            String::from_utf8_lossy(&fourcc),
            start_offset,
            start_offset + total_size,
            parent_end
        )));
    }

    Ok(Some(BoxHeader {
        fourcc,
        start_offset,
        payload_offset,
        total_size,
    }))
}

#[derive(Debug, Default)]
struct RawTrackData {
    handler_type: [u8; 4],
    tkhd_width: u32,
    tkhd_height: u32,
    timescale: u32,
    duration: u64,
    codec_fourcc: [u8; 4],
    stsd_width: u16,
    stsd_height: u16,
    bit_depth: u16,
    // Audio stsd fields
    audio_channels: u16,
    audio_sample_bits: u16,
    audio_sample_rate: u32,
    // Timecode stsd fields
    tmcd_flags: u32,
    tmcd_timescale: u32,
    tmcd_frame_duration: u32,
    tmcd_num_frames: u8,
    // Sample table (stbl) vectors
    stts_entries: Vec<(u32, u32)>,
    stsc_entries: Vec<StscEntry>,
    stsz_uniform_size: u32,
    stsz_sample_count: u32,
    stsz_entry_sizes: Vec<u32>,
    chunk_offsets: Vec<u64>,
}

#[derive(Debug, Clone, Copy)]
struct StscEntry {
    first_chunk: u32, // 1-based chunk index
    samples_per_chunk: u32,
    _sample_description_index: u32,
}

/// Opens and demuxes a Nikon `.nev` file from disk using zero-copy box walking.
pub fn demux_nev_file<P: AsRef<Path>>(path: P) -> Result<NevDemuxedContainer, DemuxError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    demux_nev_reader(&mut reader)
}

/// Demuxes any `Read + Seek` stream containing a 64-bit ISO-BMFF / QuickTime `.nev` container.
pub fn demux_nev_reader<R: Read + Seek>(reader: &mut R) -> Result<NevDemuxedContainer, DemuxError> {
    let file_len = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(0))?;

    let mut major_brand = *b"niko";
    let mut tracks: Vec<RawTrackData> = Vec::new();
    let mut ncdt_metadata: Option<NikonMetadata> = None;

    while let Some(box_hdr) = read_box_header(reader, file_len)? {
        match &box_hdr.fourcc {
            b"ftyp" => {
                if box_hdr.payload_size() >= 4 {
                    reader.read_exact(&mut major_brand)?;
                }
            }
            b"moov" => {
                parse_moov_box(
                    reader,
                    box_hdr.end_offset(),
                    &mut tracks,
                    &mut ncdt_metadata,
                )?;
            }
            b"udta" => {
                parse_udta_box(reader, box_hdr.end_offset(), &mut ncdt_metadata)?;
            }
            _ => {
                // Zero-copy skip over `mdat`, `free`, `wide`, etc.
            }
        }
        reader.seek(SeekFrom::Start(box_hdr.end_offset()))?;
    }

    // Resolve video track
    let video_track = tracks
        .iter()
        .find(|t| &t.handler_type == b"vide")
        .ok_or(DemuxError::MissingVideoTrack)?;

    let frames = resolve_video_frames(video_track)?;
    let width = if video_track.stsd_width > 0 {
        video_track.stsd_width as u32
    } else {
        video_track.tkhd_width
    };
    let height = if video_track.stsd_height > 0 {
        video_track.stsd_height as u32
    } else {
        video_track.tkhd_height
    };
    let bit_depth = if video_track.bit_depth > 0 {
        video_track.bit_depth
    } else {
        12
    };

    // Resolve optional audio track (`soun`)
    let audio_track = tracks
        .iter()
        .find(|t| &t.handler_type == b"soun")
        .and_then(|t| resolve_audio_track(t).ok());

    // Resolve optional timecode track (`tmcd`)
    let timecode_track = tracks
        .iter()
        .find(|t| &t.handler_type == b"tmcd")
        .and_then(|t| resolve_timecode_track(reader, t).ok());

    let mut metadata = ncdt_metadata.unwrap_or_default();
    if let Some(ref tc) = timecode_track {
        metadata.timecode = Some(tc.timecode_str.clone());
    }
    if bit_depth == 14 && metadata.white_level == 4095 {
        metadata.white_level = 16383;
    }

    Ok(NevDemuxedContainer {
        major_brand,
        video_codec_fourcc: video_track.codec_fourcc,
        width,
        height,
        bit_depth,
        timescale: video_track.timescale.max(1),
        duration: video_track.duration,
        frames,
        audio_track,
        timecode_track,
        metadata,
    })
}

fn parse_moov_box<R: Read + Seek>(
    reader: &mut R,
    moov_end: u64,
    tracks: &mut Vec<RawTrackData>,
    ncdt_metadata: &mut Option<NikonMetadata>,
) -> Result<(), DemuxError> {
    while let Some(hdr) = read_box_header(reader, moov_end)? {
        match &hdr.fourcc {
            b"trak" => {
                let mut track = RawTrackData::default();
                parse_trak_box(reader, hdr.end_offset(), &mut track)?;
                tracks.push(track);
            }
            b"udta" => {
                parse_udta_box(reader, hdr.end_offset(), ncdt_metadata)?;
            }
            _ => {}
        }
        reader.seek(SeekFrom::Start(hdr.end_offset()))?;
    }
    Ok(())
}

fn parse_udta_box<R: Read + Seek>(
    reader: &mut R,
    udta_end: u64,
    ncdt_metadata: &mut Option<NikonMetadata>,
) -> Result<(), DemuxError> {
    while let Some(hdr) = read_box_header(reader, udta_end)? {
        if &hdr.fourcc == b"NCDT" {
            let size = hdr.payload_size() as usize;
            // Guard against corrupt huge allocations while allowing realistic NCDT boxes (up to 16 MB)
            if size > 0 && size <= 16 * 1024 * 1024 {
                let mut payload = vec![0u8; size];
                reader.read_exact(&mut payload)?;
                if let Ok(meta) = parse_ncdt_payload(&payload) {
                    *ncdt_metadata = Some(meta);
                }
            }
        }
        reader.seek(SeekFrom::Start(hdr.end_offset()))?;
    }
    Ok(())
}

fn parse_trak_box<R: Read + Seek>(
    reader: &mut R,
    trak_end: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    while let Some(hdr) = read_box_header(reader, trak_end)? {
        match &hdr.fourcc {
            b"tkhd" => parse_tkhd_box(reader, hdr.payload_size(), track)?,
            b"mdia" => parse_mdia_box(reader, hdr.end_offset(), track)?,
            _ => {}
        }
        reader.seek(SeekFrom::Start(hdr.end_offset()))?;
    }
    Ok(())
}

fn parse_tkhd_box<R: Read + Seek>(
    reader: &mut R,
    payload_size: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    if payload_size < 84 {
        return Ok(());
    }
    let mut ver_flags = [0u8; 4];
    reader.read_exact(&mut ver_flags)?;
    let version = ver_flags[0];

    // Skip to width & height at end of tkhd:
    // version 0: 4 (ver/flags) + 2*4 (times) + 4 (track_id) + 4 (res) + 4 (dur) + 8 (res) + 2+2+2+2 + 36 (matrix) = 76 bytes before width/height
    // version 1: 4 (ver/flags) + 2*8 (times) + 4 (track_id) + 4 (res) + 8 (dur) + 8 (res) + 2+2+2+2 + 36 (matrix) = 88 bytes before width/height
    let skip_bytes = if version == 1 { 84 } else { 72 };
    if payload_size < (4 + skip_bytes + 8) {
        return Ok(());
    }
    reader.seek(SeekFrom::Current(skip_bytes as i64))?;
    let mut wh = [0u8; 8];
    reader.read_exact(&mut wh)?;
    track.tkhd_width = u32::from_be_bytes([wh[0], wh[1], wh[2], wh[3]]) >> 16;
    track.tkhd_height = u32::from_be_bytes([wh[4], wh[5], wh[6], wh[7]]) >> 16;
    Ok(())
}

fn parse_mdia_box<R: Read + Seek>(
    reader: &mut R,
    mdia_end: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    while let Some(hdr) = read_box_header(reader, mdia_end)? {
        match &hdr.fourcc {
            b"mdhd" => parse_mdhd_box(reader, hdr.payload_size(), track)?,
            b"hdlr" => parse_hdlr_box(reader, hdr.payload_size(), track)?,
            b"minf" => parse_minf_box(reader, hdr.end_offset(), track)?,
            _ => {}
        }
        reader.seek(SeekFrom::Start(hdr.end_offset()))?;
    }
    Ok(())
}

fn parse_mdhd_box<R: Read + Seek>(
    reader: &mut R,
    payload_size: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    if payload_size < 20 {
        return Ok(());
    }
    let mut ver_flags = [0u8; 4];
    reader.read_exact(&mut ver_flags)?;
    let version = ver_flags[0];

    if version == 1 && payload_size >= 28 {
        // 8 bytes creation_time + 8 bytes modification_time
        reader.seek(SeekFrom::Current(16))?;
        let mut buf = [0u8; 12];
        reader.read_exact(&mut buf)?;
        track.timescale = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        track.duration = u64::from_be_bytes([
            buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11],
        ]);
    } else {
        // 4 bytes creation_time + 4 bytes modification_time
        reader.seek(SeekFrom::Current(8))?;
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        track.timescale = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
        track.duration = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as u64;
    }
    Ok(())
}

fn parse_hdlr_box<R: Read + Seek>(
    reader: &mut R,
    payload_size: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    if payload_size < 12 {
        return Ok(());
    }
    let mut buf = [0u8; 12];
    reader.read_exact(&mut buf)?;
    // bytes 0..4: version/flags, bytes 4..8: component_type (e.g. 'mhlr'), bytes 8..12: handler_type ('vide', 'soun', 'tmcd')
    track.handler_type = [buf[8], buf[9], buf[10], buf[11]];
    Ok(())
}

fn parse_minf_box<R: Read + Seek>(
    reader: &mut R,
    minf_end: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    while let Some(hdr) = read_box_header(reader, minf_end)? {
        if &hdr.fourcc == b"stbl" {
            parse_stbl_box(reader, hdr.end_offset(), track)?;
        }
        reader.seek(SeekFrom::Start(hdr.end_offset()))?;
    }
    Ok(())
}

fn parse_stbl_box<R: Read + Seek>(
    reader: &mut R,
    stbl_end: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    while let Some(hdr) = read_box_header(reader, stbl_end)? {
        match &hdr.fourcc {
            b"stsd" => parse_stsd_box(reader, hdr.payload_size(), track)?,
            b"stts" => parse_stts_box(reader, hdr.payload_size(), track)?,
            b"stsc" => parse_stsc_box(reader, hdr.payload_size(), track)?,
            b"stsz" => parse_stsz_box(reader, hdr.payload_size(), track)?,
            b"co64" => parse_co64_box(reader, hdr.payload_size(), track)?,
            b"stco" => parse_stco_box(reader, hdr.payload_size(), track)?,
            _ => {}
        }
        reader.seek(SeekFrom::Start(hdr.end_offset()))?;
    }
    Ok(())
}

fn parse_stsd_box<R: Read + Seek>(
    reader: &mut R,
    payload_size: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    if payload_size < 16 {
        return Ok(());
    }
    if payload_size > 4 * 1024 * 1024 {
        return Err(DemuxError::InvalidContainer(
            "stsd box exceeds 4 MB safety limit".into(),
        ));
    }
    let mut buf = vec![0u8; payload_size as usize];
    reader.read_exact(&mut buf)?;

    let entry_count = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]);
    if entry_count == 0 || buf.len() < 16 {
        return Ok(());
    }

    // First sample description entry starts at offset 8
    let entry = &buf[8..];
    if entry.len() < 16 {
        return Ok(());
    }
    let codec_fourcc = [entry[4], entry[5], entry[6], entry[7]];
    track.codec_fourcc = codec_fourcc;

    match &track.handler_type {
        b"vide" => {
            // VideoSampleEntry layout:
            // 0..4: size, 4..8: fourcc, 8..14: reserved, 14..16: data_ref_index
            // 16..32: version/rev/vendor/temporal_qual/spatial_qual
            // 32..34: width (u16), 34..36: height (u16)
            // ...
            // 82..84: depth (u16)
            if entry.len() >= 36 {
                track.stsd_width = u16::from_be_bytes([entry[32], entry[33]]);
                track.stsd_height = u16::from_be_bytes([entry[34], entry[35]]);
            }
            if entry.len() >= 84 {
                let depth = u16::from_be_bytes([entry[82], entry[83]]);
                if depth == 12 || depth == 14 || depth == 16 || depth == 24 {
                    track.bit_depth = if depth == 24 { 12 } else { depth };
                }
            }
        }
        b"soun" => {
            // AudioSampleEntry layout (v0 / v1 / v2):
            // 0..16: header + data_ref_index
            // 16..18: version (u16), 18..24: revision/vendor
            // 24..26: channelcount (u16), 26..28: samplesize (u16)
            // 28..32: compression_id/packet_size
            // 32..36: samplerate (16.16 fixed point for v0/v1)
            if entry.len() >= 36 {
                let qt_version = u16::from_be_bytes([entry[16], entry[17]]);
                if qt_version == 2 && entry.len() >= 56 {
                    let sr_bits = u64::from_be_bytes([
                        entry[32], entry[33], entry[34], entry[35], entry[36], entry[37],
                        entry[38], entry[39],
                    ]);
                    track.audio_sample_rate = f64::from_bits(sr_bits).round() as u32;
                    track.audio_channels =
                        u32::from_be_bytes([entry[40], entry[41], entry[42], entry[43]]) as u16;
                    track.audio_sample_bits =
                        u32::from_be_bytes([entry[48], entry[49], entry[50], entry[51]]) as u16;
                } else {
                    track.audio_channels = u16::from_be_bytes([entry[24], entry[25]]);
                    track.audio_sample_bits = u16::from_be_bytes([entry[26], entry[27]]);
                    track.audio_sample_rate =
                        u32::from_be_bytes([entry[32], entry[33], entry[34], entry[35]]) >> 16;
                }
            }
        }
        b"tmcd" if entry.len() >= 33 => {
            // TimecodeSampleEntry layout:
            // 0..16: header + data_ref_index
            // 16..20: reserved
            // 20..24: flags (u32)
            // 24..28: timescale (u32)
            // 28..32: frame_duration (u32)
            // 32: number_of_frames (u8)
            track.tmcd_flags = u32::from_be_bytes([entry[20], entry[21], entry[22], entry[23]]);
            track.tmcd_timescale = u32::from_be_bytes([entry[24], entry[25], entry[26], entry[27]]);
            track.tmcd_frame_duration =
                u32::from_be_bytes([entry[28], entry[29], entry[30], entry[31]]);
            track.tmcd_num_frames = entry[32];
        }
        _ => {}
    }

    Ok(())
}

fn parse_stts_box<R: Read + Seek>(
    reader: &mut R,
    payload_size: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    if payload_size < 8 {
        return Ok(());
    }
    let mut hdr = [0u8; 8];
    reader.read_exact(&mut hdr)?;
    let count = u32::from_be_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
    let max_entries = ((payload_size - 8) / 8) as usize;
    let count = count.min(max_entries);

    track.stts_entries.reserve(count);
    for _ in 0..count {
        let mut entry = [0u8; 8];
        reader.read_exact(&mut entry)?;
        let sample_count = u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]);
        let sample_delta = u32::from_be_bytes([entry[4], entry[5], entry[6], entry[7]]);
        track.stts_entries.push((sample_count, sample_delta));
    }
    Ok(())
}

fn parse_stsc_box<R: Read + Seek>(
    reader: &mut R,
    payload_size: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    if payload_size < 8 {
        return Ok(());
    }
    let mut hdr = [0u8; 8];
    reader.read_exact(&mut hdr)?;
    let count = u32::from_be_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
    let max_entries = ((payload_size - 8) / 12) as usize;
    let count = count.min(max_entries);

    track.stsc_entries.reserve(count);
    for _ in 0..count {
        let mut entry = [0u8; 12];
        reader.read_exact(&mut entry)?;
        track.stsc_entries.push(StscEntry {
            first_chunk: u32::from_be_bytes([entry[0], entry[1], entry[2], entry[3]]),
            samples_per_chunk: u32::from_be_bytes([entry[4], entry[5], entry[6], entry[7]]),
            _sample_description_index: u32::from_be_bytes([
                entry[8], entry[9], entry[10], entry[11],
            ]),
        });
    }
    Ok(())
}

fn parse_stsz_box<R: Read + Seek>(
    reader: &mut R,
    payload_size: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    if payload_size < 12 {
        return Ok(());
    }
    let mut hdr = [0u8; 12];
    reader.read_exact(&mut hdr)?;
    let uniform_size = u32::from_be_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]);
    let sample_count = u32::from_be_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]);

    track.stsz_uniform_size = uniform_size;
    track.stsz_sample_count = sample_count;

    if uniform_size == 0 {
        let max_entries = ((payload_size - 12) / 4) as usize;
        let count = (sample_count as usize).min(max_entries);
        track.stsz_entry_sizes.reserve(count);
        for _ in 0..count {
            let mut b = [0u8; 4];
            reader.read_exact(&mut b)?;
            track.stsz_entry_sizes.push(u32::from_be_bytes(b));
        }
    }
    Ok(())
}

fn parse_co64_box<R: Read + Seek>(
    reader: &mut R,
    payload_size: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    if payload_size < 8 {
        return Ok(());
    }
    let mut hdr = [0u8; 8];
    reader.read_exact(&mut hdr)?;
    let count = u32::from_be_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
    let max_entries = ((payload_size - 8) / 8) as usize;
    let count = count.min(max_entries);

    track.chunk_offsets.reserve(count);
    for _ in 0..count {
        let mut b = [0u8; 8];
        reader.read_exact(&mut b)?;
        track.chunk_offsets.push(u64::from_be_bytes(b));
    }
    Ok(())
}

fn parse_stco_box<R: Read + Seek>(
    reader: &mut R,
    payload_size: u64,
    track: &mut RawTrackData,
) -> Result<(), DemuxError> {
    if payload_size < 8 || !track.chunk_offsets.is_empty() {
        return Ok(());
    }
    let mut hdr = [0u8; 8];
    reader.read_exact(&mut hdr)?;
    let count = u32::from_be_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
    let max_entries = ((payload_size - 8) / 4) as usize;
    let count = count.min(max_entries);

    track.chunk_offsets.reserve(count);
    for _ in 0..count {
        let mut b = [0u8; 4];
        reader.read_exact(&mut b)?;
        track.chunk_offsets.push(u32::from_be_bytes(b) as u64);
    }
    Ok(())
}

fn resolve_video_frames(track: &RawTrackData) -> Result<Vec<FrameDescriptor>, DemuxError> {
    if track.chunk_offsets.is_empty() {
        return Err(DemuxError::IncompleteSampleTable("co64/stco is empty"));
    }
    if track.stsc_entries.is_empty() {
        return Err(DemuxError::IncompleteSampleTable("stsc is empty"));
    }
    if track.stsz_sample_count == 0 && track.stsz_entry_sizes.is_empty() {
        return Err(DemuxError::IncompleteSampleTable("stsz has 0 samples"));
    }

    let total_samples = if track.stsz_uniform_size > 0 {
        track.stsz_sample_count as usize
    } else {
        track.stsz_entry_sizes.len()
    };

    // Expand stts run-length table into an iterator of frame durations
    let mut stts_iter = track
        .stts_entries
        .iter()
        .flat_map(|&(count, delta)| std::iter::repeat_n(delta, count as usize));

    let mut frames = Vec::with_capacity(total_samples);
    let mut sample_idx = 0usize;
    let mut current_pts = 0u64;

    let num_chunks = track.chunk_offsets.len();
    for chunk_zero_idx in 0..num_chunks {
        if sample_idx >= total_samples {
            break;
        }
        let chunk_one_based = (chunk_zero_idx + 1) as u32;

        // Find active stsc entry for this 1-based chunk index
        let samples_in_chunk = track
            .stsc_entries
            .iter()
            .rev()
            .find(|e| e.first_chunk <= chunk_one_based)
            .map(|e| e.samples_per_chunk)
            .unwrap_or(1);

        let mut offset_in_file = track.chunk_offsets[chunk_zero_idx];

        for _ in 0..samples_in_chunk {
            if sample_idx >= total_samples {
                break;
            }
            let byte_size = if track.stsz_uniform_size > 0 {
                track.stsz_uniform_size
            } else {
                track.stsz_entry_sizes[sample_idx]
            };
            let duration = stts_iter.next().unwrap_or(1001);

            frames.push(FrameDescriptor {
                frame_index: sample_idx as u32,
                file_offset: offset_in_file,
                byte_size,
                pts: current_pts,
                duration,
            });

            offset_in_file += byte_size as u64;
            current_pts += duration as u64;
            sample_idx += 1;
        }
    }

    Ok(frames)
}

fn resolve_audio_track(track: &RawTrackData) -> Result<AudioTrackInfo, DemuxError> {
    if track.chunk_offsets.is_empty() || track.stsc_entries.is_empty() {
        return Err(DemuxError::IncompleteSampleTable("audio stbl incomplete"));
    }

    let channels = track.audio_channels.max(1);
    let sample_bits = if track.audio_sample_bits > 0 {
        track.audio_sample_bits
    } else {
        24
    };
    let bytes_per_frame = (channels as u32) * ((sample_bits as u32) / 8).max(1);

    let mut chunks = Vec::with_capacity(track.chunk_offsets.len());
    let mut sample_cursor = 0usize;
    let mut total_samples = 0u64;

    for (chunk_zero_idx, &file_offset) in track.chunk_offsets.iter().enumerate() {
        let chunk_one_based = (chunk_zero_idx + 1) as u32;
        let samples_in_chunk = track
            .stsc_entries
            .iter()
            .rev()
            .find(|e| e.first_chunk <= chunk_one_based)
            .map(|e| e.samples_per_chunk)
            .unwrap_or(1);

        let byte_size = if track.stsz_uniform_size > 1 {
            samples_in_chunk * track.stsz_uniform_size
        } else if track.stsz_uniform_size == 1 {
            // In QuickTime PCM stsz, sample_size == 1 often means 1 PCM sample frame
            samples_in_chunk * bytes_per_frame
        } else {
            let mut sum = 0u32;
            for _ in 0..samples_in_chunk {
                if let Some(&sz) = track.stsz_entry_sizes.get(sample_cursor) {
                    sum += sz;
                    sample_cursor += 1;
                }
            }
            sum
        };

        chunks.push(AudioChunkDescriptor {
            file_offset,
            byte_size,
            sample_count: samples_in_chunk,
        });
        total_samples += samples_in_chunk as u64;
    }

    Ok(AudioTrackInfo {
        codec_fourcc: track.codec_fourcc,
        channels,
        sample_bits,
        sample_rate: track.audio_sample_rate.max(48000),
        timescale: track.timescale.max(48000),
        total_samples,
        chunks,
    })
}

fn resolve_timecode_track<R: Read + Seek>(
    reader: &mut R,
    track: &RawTrackData,
) -> Result<TimecodeTrackInfo, DemuxError> {
    let first_offset = *track
        .chunk_offsets
        .first()
        .ok_or(DemuxError::IncompleteSampleTable("tmcd chunk offset"))?;

    reader.seek(SeekFrom::Start(first_offset))?;
    let mut tc_bytes = [0u8; 4];
    reader.read_exact(&mut tc_bytes)?;
    let frame_number = u32::from_be_bytes(tc_bytes);

    let num_frames = track.tmcd_num_frames.max(24) as u32;
    let ff = frame_number % num_frames;
    let total_seconds = frame_number / num_frames;
    let ss = total_seconds % 60;
    let mm = (total_seconds / 60) % 60;
    let hh = total_seconds / 3600;

    let drop_frame = (track.tmcd_flags & 0x0001) != 0;
    let sep = if drop_frame { ';' } else { ':' };
    let timecode_str = format!("{:02}:{:02}:{:02}{}{:02}", hh, mm, ss, sep, ff);

    Ok(TimecodeTrackInfo {
        flags: track.tmcd_flags,
        timescale: track.tmcd_timescale.max(track.timescale).max(24000),
        frame_duration: track.tmcd_frame_duration.max(1001),
        num_frames: num_frames as u8,
        frame_number,
        timecode_str,
    })
}

// ============================================================================
// Synthetic 64-Bit `.nev` Builder (For Testing & Self-Contained Verification)
// ============================================================================

/// Builds a valid 64-bit ISO-BMFF Nikon `.nev` container in memory with:
/// - `ftyp` (`niko`)
/// - 64-bit extended `mdat` (`size == 1`) containing TicoRAW RGGB frames, 24-bit PCM audio, and `tmcd` sample
/// - `moov` with Video (`vide`), Audio (`soun`), Timecode (`tmcd`) tracks using `co64`, and `udta -> NCDT` MakerNote
pub fn build_synthetic_nev(
    width: u16,
    height: u16,
    frames_bayer: &[Vec<u16>],
    wb_rb_levels: [f32; 4],
    black_level: [u16; 4],
    iso: u16,
) -> Vec<u8> {
    build_synthetic_nev_with_bit_depth(
        width,
        height,
        12,
        frames_bayer,
        wb_rb_levels,
        black_level,
        iso,
    )
}

/// Builds a valid 64-bit ISO-BMFF Nikon `.nev` container in memory with configurable sensor `bit_depth` (12 or 14).
pub fn build_synthetic_nev_with_bit_depth(
    width: u16,
    height: u16,
    bit_depth: u16,
    frames_bayer: &[Vec<u16>],
    wb_rb_levels: [f32; 4],
    black_level: [u16; 4],
    iso: u16,
) -> Vec<u8> {
    let mut out = Vec::new();

    // 1. `ftyp` box (`niko`)
    let mut ftyp_payload = Vec::new();
    ftyp_payload.extend_from_slice(b"niko");
    ftyp_payload.extend_from_slice(&0x0000_0200u32.to_be_bytes());
    ftyp_payload.extend_from_slice(b"nikoqt  ");
    write_box32(&mut out, b"ftyp", &ftyp_payload);

    // 2. Prepare `mdat` payload with 64-bit extended header (`size == 1`)
    let mdat_start = out.len();
    out.extend_from_slice(&1u32.to_be_bytes()); // size == 1 signals 64-bit extended size
    out.extend_from_slice(b"mdat");
    out.extend_from_slice(&0u64.to_be_bytes()); // placeholder for 64-bit size

    let mut video_offsets = Vec::new();
    let mut video_sizes = Vec::new();
    for frame in frames_bayer {
        let offset = out.len() as u64;
        let encoded = if bit_depth >= 14 {
            crate::decoder::encode_ticoraw_14bit_test_frame(frame, width as usize, height as usize)
        } else {
            crate::decoder::encode_ticoraw_test_frame(frame, width as usize, height as usize)
        };
        video_offsets.push(offset);
        video_sizes.push(encoded.len() as u32);
        out.extend_from_slice(&encoded);
    }

    // Write 1 chunk of 24-bit stereo 48kHz Linear PCM audio (2000 samples = 12000 bytes)
    let audio_offset = out.len() as u64;
    let audio_samples = 2000u32;
    let audio_bytes_len = audio_samples * 2 * 3; // 2 channels * 3 bytes (24-bit)
    for i in 0..(audio_bytes_len as usize / 3) {
        let sample_val = ((i as i32 * 97) & 0x7FFFFF) - 0x400000;
        let b = sample_val.to_le_bytes();
        out.extend_from_slice(&[b[0], b[1], b[2]]);
    }

    // Write 1 timecode sample (`01:00:00:00` at 24 fps = 86400 frames)
    let tmcd_offset = out.len() as u64;
    let tc_frames: u32 = 86_400;
    out.extend_from_slice(&tc_frames.to_be_bytes());

    // Patch 64-bit `mdat` size
    let mdat_total_size = (out.len() - mdat_start) as u64;
    out[mdat_start + 8..mdat_start + 16].copy_from_slice(&mdat_total_size.to_be_bytes());

    // 3. Build `moov` box
    let mut moov = Vec::new();

    // 3a. Video `trak`
    let video_trak = build_test_trak(
        1,
        b"vide",
        b"traw",
        width,
        height,
        bit_depth,
        24000,
        (frames_bayer.len() as u64) * 1001,
        &[(frames_bayer.len() as u32, 1001)],
        &[(1, 1, 1)],
        0,
        &video_sizes,
        &video_offsets,
    );
    write_box32(&mut moov, b"trak", &video_trak);

    // 3b. Audio `trak`
    let audio_trak = build_test_trak(
        2,
        b"soun",
        b"in24",
        0,
        0,
        24,
        48000,
        audio_samples as u64,
        &[(audio_samples, 1)],
        &[(1, audio_samples, 1)],
        6, // 2 channels * 3 bytes per sample
        &[],
        &[audio_offset],
    );
    write_box32(&mut moov, b"trak", &audio_trak);

    // 3c. Timecode `trak`
    let tmcd_trak = build_test_trak(
        3,
        b"tmcd",
        b"tmcd",
        0,
        0,
        0,
        24000,
        (frames_bayer.len() as u64) * 1001,
        &[(1, (frames_bayer.len() as u32) * 1001)],
        &[(1, 1, 1)],
        4,
        &[],
        &[tmcd_offset],
    );
    write_box32(&mut moov, b"trak", &tmcd_trak);

    // 3d. `udta -> NCDT` with Nikon MakerNote
    let ncdt_payload = build_test_ncdt_makernote(wb_rb_levels, black_level, iso);
    let mut udta_payload = Vec::new();
    write_box32(&mut udta_payload, b"NCDT", &ncdt_payload);
    write_box32(&mut moov, b"udta", &udta_payload);

    write_box32(&mut out, b"moov", &moov);
    out
}

fn write_box32(out: &mut Vec<u8>, fourcc: &[u8; 4], payload: &[u8]) {
    let size = (8 + payload.len()) as u32;
    out.extend_from_slice(&size.to_be_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(payload);
}

#[allow(clippy::too_many_arguments)]
fn build_test_trak(
    track_id: u32,
    handler: &[u8; 4],
    codec: &[u8; 4],
    width: u16,
    height: u16,
    bit_depth: u16,
    timescale: u32,
    duration: u64,
    stts: &[(u32, u32)],
    stsc: &[(u32, u32, u32)],
    uniform_size: u32,
    entry_sizes: &[u32],
    co64_offsets: &[u64],
) -> Vec<u8> {
    let mut trak = Vec::new();

    // tkhd (version 0, 84 bytes payload)
    let mut tkhd = vec![0u8; 84];
    tkhd[0] = 0; // version 0
    tkhd[3] = 0x03; // track enabled
    tkhd[12..16].copy_from_slice(&track_id.to_be_bytes());
    tkhd[20..24].copy_from_slice(&(duration as u32).to_be_bytes());
    tkhd[76..80].copy_from_slice(&((width as u32) << 16).to_be_bytes());
    tkhd[80..84].copy_from_slice(&((height as u32) << 16).to_be_bytes());
    write_box32(&mut trak, b"tkhd", &tkhd);

    // mdia
    let mut mdia = Vec::new();
    let mut mdhd = vec![0u8; 24];
    mdhd[12..16].copy_from_slice(&timescale.to_be_bytes());
    mdhd[16..20].copy_from_slice(&(duration as u32).to_be_bytes());
    write_box32(&mut mdia, b"mdhd", &mdhd);

    let mut hdlr = vec![0u8; 24];
    hdlr[4..8].copy_from_slice(b"mhlr");
    hdlr[8..12].copy_from_slice(handler);
    write_box32(&mut mdia, b"hdlr", &hdlr);

    // minf -> stbl
    let mut stbl = Vec::new();

    // stsd
    let mut stsd = Vec::new();
    stsd.extend_from_slice(&0u32.to_be_bytes());
    stsd.extend_from_slice(&1u32.to_be_bytes()); // 1 entry
    let mut entry = vec![0u8; 86];
    let entry_len = entry.len() as u32;
    entry[0..4].copy_from_slice(&entry_len.to_be_bytes());
    entry[4..8].copy_from_slice(codec);
    entry[14..16].copy_from_slice(&1u16.to_be_bytes());
    if handler == b"vide" {
        entry[32..34].copy_from_slice(&width.to_be_bytes());
        entry[34..36].copy_from_slice(&height.to_be_bytes());
        entry[82..84].copy_from_slice(&bit_depth.to_be_bytes());
    } else if handler == b"soun" {
        entry[24..26].copy_from_slice(&2u16.to_be_bytes()); // 2 channels
        entry[26..28].copy_from_slice(&24u16.to_be_bytes()); // 24-bit
        entry[32..36].copy_from_slice(&(48000u32 << 16).to_be_bytes());
    } else if handler == b"tmcd" {
        entry[20..24].copy_from_slice(&0u32.to_be_bytes()); // non-drop
        entry[24..28].copy_from_slice(&24000u32.to_be_bytes());
        entry[28..32].copy_from_slice(&1001u32.to_be_bytes());
        entry[32] = 24;
    }
    stsd.extend_from_slice(&entry);
    write_box32(&mut stbl, b"stsd", &stsd);

    // stts
    let mut stts_box = Vec::new();
    stts_box.extend_from_slice(&0u32.to_be_bytes());
    stts_box.extend_from_slice(&(stts.len() as u32).to_be_bytes());
    for &(c, d) in stts {
        stts_box.extend_from_slice(&c.to_be_bytes());
        stts_box.extend_from_slice(&d.to_be_bytes());
    }
    write_box32(&mut stbl, b"stts", &stts_box);

    // stsc
    let mut stsc_box = Vec::new();
    stsc_box.extend_from_slice(&0u32.to_be_bytes());
    stsc_box.extend_from_slice(&(stsc.len() as u32).to_be_bytes());
    for &(fc, spc, sdi) in stsc {
        stsc_box.extend_from_slice(&fc.to_be_bytes());
        stsc_box.extend_from_slice(&spc.to_be_bytes());
        stsc_box.extend_from_slice(&sdi.to_be_bytes());
    }
    write_box32(&mut stbl, b"stsc", &stsc_box);

    // stsz
    let mut stsz_box = Vec::new();
    stsz_box.extend_from_slice(&0u32.to_be_bytes());
    stsz_box.extend_from_slice(&uniform_size.to_be_bytes());
    let sample_count = if uniform_size > 0 {
        stts.iter().map(|&(c, _)| c).sum()
    } else {
        entry_sizes.len() as u32
    };
    stsz_box.extend_from_slice(&sample_count.to_be_bytes());
    if uniform_size == 0 {
        for &sz in entry_sizes {
            stsz_box.extend_from_slice(&sz.to_be_bytes());
        }
    }
    write_box32(&mut stbl, b"stsz", &stsz_box);

    // co64
    let mut co64_box = Vec::new();
    co64_box.extend_from_slice(&0u32.to_be_bytes());
    co64_box.extend_from_slice(&(co64_offsets.len() as u32).to_be_bytes());
    for &off in co64_offsets {
        co64_box.extend_from_slice(&off.to_be_bytes());
    }
    write_box32(&mut stbl, b"co64", &co64_box);

    let mut minf = Vec::new();
    write_box32(&mut minf, b"stbl", &stbl);
    write_box32(&mut mdia, b"minf", &minf);
    write_box32(&mut trak, b"mdia", &mdia);

    trak
}

fn build_test_ncdt_makernote(wb_rb_levels: [f32; 4], black_level: [u16; 4], iso: u16) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(b"Nikon\0");
    buf.extend_from_slice(&[0x02, 0x10, 0x00, 0x00]);
    let tiff_base = buf.len();
    buf.extend_from_slice(b"II\x2a\x00");
    buf.extend_from_slice(&8u32.to_le_bytes());

    let num_entries = 4u16;
    buf.extend_from_slice(&num_entries.to_le_bytes());
    let data_offset = 8 + 2 + (num_entries as usize) * 12 + 4;

    // 1. Tag 0x0002 (ISO)
    buf.extend_from_slice(&0x0002u16.to_le_bytes());
    buf.extend_from_slice(&3u16.to_le_bytes());
    buf.extend_from_slice(&2u32.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes());
    buf.extend_from_slice(&iso.to_le_bytes());

    // 2. Tag 0x000C (WB_RBLevels)
    buf.extend_from_slice(&0x000Cu16.to_le_bytes());
    buf.extend_from_slice(&5u16.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&(data_offset as u32).to_le_bytes());

    // 3. Tag 0x003D (BlackLevel)
    buf.extend_from_slice(&0x003Du16.to_le_bytes());
    buf.extend_from_slice(&3u16.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&((data_offset + 32) as u32).to_le_bytes());

    // 4. Tag 0x829A (ExposureTime = 1/48s)
    buf.extend_from_slice(&0x829Au16.to_le_bytes());
    buf.extend_from_slice(&5u16.to_le_bytes());
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&((data_offset + 40) as u32).to_le_bytes());

    buf.extend_from_slice(&0u32.to_le_bytes());
    assert_eq!(buf.len() - tiff_base, data_offset);

    for &g in &wb_rb_levels {
        let num = (g * 10000.0).round() as u32;
        buf.extend_from_slice(&num.to_le_bytes());
        buf.extend_from_slice(&10000u32.to_le_bytes());
    }
    for &b in &black_level {
        buf.extend_from_slice(&b.to_le_bytes());
    }
    buf.extend_from_slice(&1u32.to_le_bytes());
    buf.extend_from_slice(&48u32.to_le_bytes());

    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_demux_synthetic_64bit_nev() {
        let w = 64u16;
        let h = 36u16;
        let frame0 = vec![1024u16; (w as usize) * (h as usize)];
        let frame1 = vec![2048u16; (w as usize) * (h as usize)];
        let nev_bytes = build_synthetic_nev(
            w,
            h,
            &[frame0, frame1],
            [2.05, 1.48, 1.0, 1.0],
            [150, 150, 150, 150],
            800,
        );

        let mut cursor = Cursor::new(nev_bytes);
        let container = demux_nev_reader(&mut cursor).expect("should demux synthetic .nev");

        assert_eq!(&container.major_brand, b"niko");
        assert_eq!(container.width, 64);
        assert_eq!(container.height, 36);
        assert_eq!(container.frames.len(), 2);
        assert_eq!(container.frames[0].pts, 0);
        assert_eq!(container.frames[1].pts, 1001);
        assert_eq!(container.metadata.iso_speed, 800);
        assert_eq!(container.metadata.black_level, [150, 150, 150, 150]);
        assert!((container.metadata.wb_rb_levels[0] - 2.05).abs() < 1e-3);
        assert_eq!(container.metadata.timecode.as_deref(), Some("01:00:00:00"));
        assert!(container.audio_track.is_some());
    }

    #[test]
    fn test_demux_malformed_and_truncated_containers_never_panic() {
        // Empty input -> MissingVideoTrack
        let mut empty = Cursor::new(Vec::<u8>::new());
        assert!(matches!(
            demux_nev_reader(&mut empty),
            Err(DemuxError::MissingVideoTrack)
        ));

        // Invalid 32-bit box size (< 8 bytes)
        let bad_size32 = [0, 0, 0, 4, b'm', b'o', b'o', b'v'];
        assert!(matches!(
            demux_nev_reader(&mut Cursor::new(bad_size32)),
            Err(DemuxError::InvalidContainer(_))
        ));

        // Truncated 64-bit extended box header (size == 1 without 8-byte largesize)
        let trunc_size64 = [0, 0, 0, 1, b'm', b'd', b'a', b't', 0, 0, 0];
        assert!(matches!(
            demux_nev_reader(&mut Cursor::new(trunc_size64)),
            Err(DemuxError::InvalidContainer(_))
        ));

        // Box size exceeding file boundary (OOM / out-of-bounds guard)
        let mut huge_box = Vec::new();
        huge_box.extend_from_slice(&1u32.to_be_bytes());
        huge_box.extend_from_slice(b"moov");
        huge_box.extend_from_slice(&u64::MAX.to_be_bytes());
        assert!(matches!(
            demux_nev_reader(&mut Cursor::new(huge_box)),
            Err(DemuxError::InvalidContainer(_))
        ));
    }
}
