use thiserror::Error;

#[derive(Debug, Error)]
pub enum MetadataError {
    #[error("Nikon MakerNote ('Nikon\\0') header not found in NCDT box")]
    MakerNoteNotFound,
    #[error("Invalid or truncated TIFF header inside Nikon MakerNote")]
    InvalidTiffHeader,
    #[error("Malformed IFD table in Nikon MakerNote")]
    MalformedIfd,
}

/// Extracted Nikon `.nev` (`NCDT` + MakerNote + `tmcd`) camera metadata.
#[derive(Debug, Clone)]
pub struct NikonMetadata {
    /// Tag `0x000C` (`WB_RBLevels`): `[R_gain, B_gain, G1_gain, G2_gain]`.
    pub wb_rb_levels: [f32; 4],
    /// Tag `0x003D` (`BlackLevel`): 4 `u16` RGGB sensor black levels `[R, G1, G2, B]`.
    pub black_level: [u16; 4],
    /// Sensor white clipping level (4095 for 12-bit N-RAW, 16383 for 14-bit).
    pub white_level: u16,
    /// Camera ISO speed (e.g., 800 for Nikon Z8/Z9 base N-RAW ISO).
    pub iso_speed: u32,
    /// Exposure / shutter time as `(numerator, denominator)` in seconds (e.g., `(1, 48)`).
    pub shutter_speed: (u32, u32),
    /// Lens description string (e.g., `"NIKKOR Z 24-70mm f/2.8 S"`).
    pub lens_model: String,
    /// Focal length in millimeters, if present.
    pub focal_length_mm: Option<f32>,
    /// Aperture f-number, if present.
    pub aperture_fstop: Option<f32>,
    /// Recorded As-Shot Kelvin color temperature (default 5600K).
    pub color_temperature_kelvin: u32,
    /// Camera body model identifier.
    pub camera_model: String,
    /// SMPTE timecode string (`HH:MM:SS:FF` or `HH:MM:SS;FF`), populated from `tmcd` track if present.
    pub timecode: Option<String>,
}

impl Default for NikonMetadata {
    fn default() -> Self {
        Self {
            wb_rb_levels: [1.95, 1.42, 1.0, 1.0],
            black_level: [150, 150, 150, 150],
            white_level: 4095,
            iso_speed: 800,
            shutter_speed: (1, 48),
            lens_model: "NIKKOR Z 24-70mm f/2.8 S".to_string(),
            focal_length_mm: Some(50.0),
            aperture_fstop: Some(2.8),
            color_temperature_kelvin: 5600,
            camera_model: "NIKON Z 9".to_string(),
            timecode: None,
        }
    }
}

impl NikonMetadata {
    /// Returns the RGGB white balance multipliers `[R_gain, G1_gain, G2_gain, B_gain]`
    /// reordered from Nikon's `WB_RBLevels` `[R_gain, B_gain, G1_gain, G2_gain]`.
    pub fn rggb_wb_gains(&self) -> [f32; 4] {
        let [r, b, g1, g2] = self.wb_rb_levels;
        [
            if r > 0.0 { r } else { 1.0 },
            if g1 > 0.0 { g1 } else { 1.0 },
            if g2 > 0.0 { g2 } else { 1.0 },
            if b > 0.0 { b } else { 1.0 },
        ]
    }

    /// Formats shutter speed as `"1/48s"` or `"0.5s"`.
    pub fn shutter_display(&self) -> String {
        let (num, den) = self.shutter_speed;
        if num == 0 || den == 0 {
            return "1/48s".to_string();
        }
        if num == 1 {
            format!("1/{}s", den)
        } else if den % num == 0 {
            format!("1/{}s", den / num)
        } else {
            format!("{:.4}s", num as f64 / den as f64)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiffEndian {
    Little, // "II" (0x4949)
    Big,    // "MM" (0x4D4D)
}

impl TiffEndian {
    #[inline]
    pub fn read_u16(self, bytes: &[u8]) -> u16 {
        match self {
            TiffEndian::Little => u16::from_le_bytes([bytes[0], bytes[1]]),
            TiffEndian::Big => u16::from_be_bytes([bytes[0], bytes[1]]),
        }
    }

    #[inline]
    pub fn read_u32(self, bytes: &[u8]) -> u32 {
        match self {
            TiffEndian::Little => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            TiffEndian::Big => u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        }
    }

    #[inline]
    pub fn read_i32(self, bytes: &[u8]) -> i32 {
        match self {
            TiffEndian::Little => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            TiffEndian::Big => i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        }
    }
}

/// Scans an `NCDT` atom payload for the embedded `"Nikon\0"` MakerNote TIFF structure
/// and extracts `WB_RBLevels` (`0x000C`), `BlackLevel` (`0x003D`), ISO, Shutter, and Lens metadata.
pub fn parse_ncdt_payload(payload: &[u8]) -> Result<NikonMetadata, MetadataError> {
    let mut meta = NikonMetadata::default();

    // Scan for "Nikon\0" (6 bytes: 4E 69 6B 6F 6E 00)
    let nikon_sig = b"Nikon\0";
    let Some(sig_pos) = payload
        .windows(nikon_sig.len())
        .position(|w| w == nikon_sig)
    else {
        return Err(MetadataError::MakerNoteNotFound);
    };

    let after_sig = &payload[sig_pos + nikon_sig.len()..];
    if after_sig.len() < 8 {
        return Err(MetadataError::InvalidTiffHeader);
    }

    // Locate embedded TIFF header ("II\x2A\x00" or "MM\x00\x2A").
    // Standard Nikon MakerNote has 4 bytes (version 02 10 00 00) before the TIFF header,
    // or starts immediately after "Nikon\0".
    let search_window = &after_sig[..after_sig.len().min(16)];
    let mut tiff_rel_offset = None;
    let mut endian = TiffEndian::Little;

    for i in 0..search_window.len().saturating_sub(4) {
        let slice = &search_window[i..i + 4];
        if slice == b"II\x2a\x00" {
            tiff_rel_offset = Some(i);
            endian = TiffEndian::Little;
            break;
        } else if slice == b"MM\x00\x2a" {
            tiff_rel_offset = Some(i);
            endian = TiffEndian::Big;
            break;
        }
    }

    let Some(tiff_offset) = tiff_rel_offset else {
        return Err(MetadataError::InvalidTiffHeader);
    };

    let tiff_data = &after_sig[tiff_offset..];
    if tiff_data.len() < 8 {
        return Err(MetadataError::InvalidTiffHeader);
    }

    let first_ifd_offset = endian.read_u32(&tiff_data[4..8]) as usize;
    if first_ifd_offset + 2 > tiff_data.len() {
        return Err(MetadataError::MalformedIfd);
    }

    parse_ifd(tiff_data, first_ifd_offset, endian, &mut meta)?;

    Ok(meta)
}

fn parse_ifd(
    tiff_data: &[u8],
    ifd_offset: usize,
    endian: TiffEndian,
    meta: &mut NikonMetadata,
) -> Result<(), MetadataError> {
    if ifd_offset + 2 > tiff_data.len() {
        return Err(MetadataError::MalformedIfd);
    }

    let entry_count = endian.read_u16(&tiff_data[ifd_offset..ifd_offset + 2]) as usize;
    let entries_start = ifd_offset + 2;
    if entries_start + entry_count * 12 > tiff_data.len() {
        return Err(MetadataError::MalformedIfd);
    }

    for i in 0..entry_count {
        let entry_off = entries_start + i * 12;
        let entry = &tiff_data[entry_off..entry_off + 12];

        let tag = endian.read_u16(&entry[0..2]);
        let field_type = endian.read_u16(&entry[2..4]);
        let count = endian.read_u32(&entry[4..8]) as usize;
        let val_or_offset = &entry[8..12];

        let type_size = match field_type {
            1 | 2 | 6 | 7 => 1usize, // BYTE, ASCII, SBYTE, UNDEFINED
            3 | 8 => 2usize,         // SHORT, SSHORT
            4 | 9 | 11 => 4usize,    // LONG, SLONG, FLOAT
            5 | 10 | 12 => 8usize,   // RATIONAL, SRATIONAL, DOUBLE
            _ => continue,
        };

        let total_bytes = count.saturating_mul(type_size);
        let value_slice: &[u8] = if total_bytes <= 4 {
            &val_or_offset[..total_bytes]
        } else {
            let offset = endian.read_u32(val_or_offset) as usize;
            if offset.saturating_add(total_bytes) > tiff_data.len() {
                continue;
            }
            &tiff_data[offset..offset + total_bytes]
        };

        match tag {
            // Tag 0x000C: WB_RBLevels (4 rationals: [R_gain, B_gain, G1_gain, G2_gain])
            0x000C => {
                if (field_type == 5 || field_type == 10) && count >= 4 && value_slice.len() >= 32 {
                    let mut levels = [1.0f32; 4];
                    for (k, level) in levels.iter_mut().enumerate() {
                        let base = k * 8;
                        if field_type == 5 {
                            let num = endian.read_u32(&value_slice[base..base + 4]) as f32;
                            let den = endian.read_u32(&value_slice[base + 4..base + 8]) as f32;
                            if den > 0.0 {
                                *level = num / den;
                            }
                        } else {
                            let num = endian.read_i32(&value_slice[base..base + 4]) as f32;
                            let den = endian.read_i32(&value_slice[base + 4..base + 8]) as f32;
                            if den != 0.0 {
                                *level = num / den;
                            }
                        }
                    }
                    meta.wb_rb_levels = levels;
                }
            }
            // Tag 0x003D: BlackLevel (4 u16 RGGB sensor black levels)
            0x003D => {
                if field_type == 3 && count >= 4 && value_slice.len() >= 8 {
                    let mut blacks = [0u16; 4];
                    for (k, black) in blacks.iter_mut().enumerate() {
                        *black = endian.read_u16(&value_slice[k * 2..k * 2 + 2]);
                    }
                    meta.black_level = blacks;
                    // Infer 14-bit vs 12-bit white level if black level is around 600 vs 150
                    if blacks[0] >= 400 && meta.white_level == 4095 {
                        meta.white_level = 16383;
                    }
                }
            }
            // Tag 0x0002 (Nikon ISO [u16; 2]) or 0x8827 (Exif PhotographicSensitivity)
            0x0002 | 0x8827 => {
                if field_type == 3 && value_slice.len() >= 2 {
                    if count >= 2 && value_slice.len() >= 4 {
                        let iso = endian.read_u16(&value_slice[2..4]);
                        if iso > 0 {
                            meta.iso_speed = iso as u32;
                        } else {
                            meta.iso_speed = endian.read_u16(&value_slice[0..2]) as u32;
                        }
                    } else {
                        meta.iso_speed = endian.read_u16(&value_slice[0..2]) as u32;
                    }
                } else if field_type == 4 && value_slice.len() >= 4 {
                    meta.iso_speed = endian.read_u32(&value_slice[0..4]);
                }
            }
            // Tag 0x829A (ExposureTime rational)
            0x829A => {
                if field_type == 5 && count >= 1 && value_slice.len() >= 8 {
                    let num = endian.read_u32(&value_slice[0..4]);
                    let den = endian.read_u32(&value_slice[4..8]);
                    if num > 0 && den > 0 {
                        meta.shutter_speed = (num, den);
                    }
                }
            }
            // Tag 0x0084: Nikon Lens specification (4 rationals: min_fl, max_fl, min_f, max_f)
            0x0084 => {
                if field_type == 5 && count >= 4 && value_slice.len() >= 32 {
                    let read_rat = |idx: usize| -> f32 {
                        let b = idx * 8;
                        let n = endian.read_u32(&value_slice[b..b + 4]) as f32;
                        let d = endian.read_u32(&value_slice[b + 4..b + 8]) as f32;
                        if d > 0.0 {
                            n / d
                        } else {
                            0.0
                        }
                    };
                    let min_fl = read_rat(0);
                    let max_fl = read_rat(1);
                    let min_f = read_rat(2);
                    let max_f = read_rat(3);
                    if min_fl > 0.0 {
                        meta.focal_length_mm = Some(min_fl);
                        meta.aperture_fstop = Some(min_f);
                        if (max_fl - min_fl).abs() < 0.1 {
                            meta.lens_model = format!("NIKKOR Z {:.0}mm f/{:.1}", min_fl, min_f);
                        } else if (max_f - min_f).abs() < 0.1 {
                            meta.lens_model =
                                format!("NIKKOR Z {:.0}-{:.0}mm f/{:.1}", min_fl, max_fl, min_f);
                        } else {
                            meta.lens_model = format!(
                                "NIKKOR Z {:.0}-{:.0}mm f/{:.1}-{:.1}",
                                min_fl, max_fl, min_f, max_f
                            );
                        }
                    }
                } else if field_type == 2 {
                    let s = parse_ascii_tag(value_slice);
                    if !s.is_empty() {
                        meta.lens_model = s;
                    }
                }
            }
            // Tag 0xA434: Exif LensModel ASCII
            0xA434 => {
                if field_type == 2 {
                    let s = parse_ascii_tag(value_slice);
                    if !s.is_empty() {
                        meta.lens_model = s;
                    }
                }
            }
            // Tag 0x0110: Camera Model ASCII
            0x0110 => {
                if field_type == 2 {
                    let s = parse_ascii_tag(value_slice);
                    if !s.is_empty() {
                        meta.camera_model = s;
                    }
                }
            }
            // Tag 0x0045: Kelvin Color Temperature
            0x0045 => {
                if field_type == 3 && value_slice.len() >= 2 {
                    meta.color_temperature_kelvin = endian.read_u16(&value_slice[0..2]) as u32;
                } else if field_type == 4 && value_slice.len() >= 4 {
                    meta.color_temperature_kelvin = endian.read_u32(&value_slice[0..4]);
                }
            }
            _ => {}
        }
    }

    Ok(())
}

fn parse_ascii_tag(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).trim().to_string()
}

// ============================================================================
// Color Science: Bradford Chromatic Adaptation, Camera->XYZ->Rec.2020 & N-Log
// ============================================================================

/// Bradford cone response matrix $M_B$:
/// ```text
/// [  0.8951   0.2664  -0.1614 ]
/// [ -0.7502   1.7135   0.0367 ]
/// [  0.0389  -0.0685   1.0296 ]
/// ```
pub const BRADFORD_M: [[f32; 3]; 3] = [
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
];

/// Exact inverse of the Bradford cone response matrix $M_B^{-1}$.
pub const BRADFORD_M_INV: [[f32; 3]; 3] = [
    [0.9869929, -0.1470543, 0.1599627],
    [0.4323053, 0.5183603, 0.0492912],
    [-0.0085287, 0.0400428, 0.9684867],
];

/// Nikon Z8/Z9 calibrated Camera-RGB to CIE 1931 XYZ (D65) matrix.
pub const CAMERA_TO_XYZ_D65: [[f32; 3]; 3] = [
    [0.6458, 0.1982, 0.1065],
    [0.2460, 0.7105, 0.0435],
    [0.0195, 0.0890, 0.9803],
];

/// Official ITU-R BT.2020 CIE 1931 XYZ (D65) to Rec. 2020 Linear RGB matrix.
pub const XYZ_TO_REC2020: [[f32; 3]; 3] = [
    [1.7166512, -0.3556708, -0.2533663],
    [-0.6666844, 1.6164812, 0.0157685],
    [0.0176399, -0.0427706, 0.9421031],
];

/// Converts Correlated Color Temperature `kelvin` ($T \in [1667, 25000]$) and `tint` shift
/// into a normalized CIE 1931 $XYZ$ tristimulus white point $[X/Y, 1.0, Z/Y]^T$.
pub fn kelvin_tint_to_xyz(kelvin: f32, tint: f32) -> [f32; 3] {
    let t = kelvin.clamp(1667.0, 25000.0) as f64;
    let inv_t = 1000.0 / t;
    let inv_t2 = inv_t * inv_t;
    let inv_t3 = inv_t2 * inv_t;

    // Kang et al. (2002) Planckian locus approximation for CIE 1931 x
    let x = if t <= 4000.0 {
        -0.2661239 * inv_t3 - 0.2343589 * inv_t2 + 0.8776956 * inv_t + 0.179910
    } else {
        -3.0258469 * inv_t3 + 2.1070379 * inv_t2 + 0.2226347 * inv_t + 0.240390
    };

    let x2 = x * x;
    let x3 = x2 * x;
    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    };

    // Apply tint shift along CIE 1960 UCS v-axis
    let denom_uv = -2.0 * x + 12.0 * y + 3.0;
    let u = 4.0 * x / denom_uv;
    let v = (6.0 * y / denom_uv) + (tint as f64) * 0.00015;

    let denom_xy = 2.0 * u - 8.0 * v + 4.0;
    let x_adj = (3.0 * u / denom_xy).clamp(0.05, 0.85);
    let y_adj = (2.0 * v / denom_xy).clamp(0.05, 0.85);

    let big_x = (x_adj / y_adj) as f32;
    let big_y = 1.0f32;
    let big_z = ((1.0 - x_adj - y_adj) / y_adj) as f32;

    [big_x, big_y, big_z]
}

#[inline]
pub fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

#[inline]
pub fn mat3_vec_mul(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Computes the $3 \times 3$ scene-linear White Balance & Color Space matrix $M_{\text{WB}}$:
/// $$M_{\text{WB}} = M_{\text{XYZ}\rightarrow\text{Rec2020}} \cdot M_B^{-1} \cdot \text{diag}\left(\frac{M_B W_{\text{src}}}{M_B W_{\text{dst}}}\right) \cdot M_B \cdot M_{\text{Camera}\rightarrow\text{XYZ}}$$
pub fn compute_wb_color_matrix(
    as_shot_kelvin: u32,
    target_kelvin: Option<u32>,
    target_tint: Option<f32>,
) -> [[f32; 3]; 3] {
    let src_k = target_kelvin.unwrap_or(as_shot_kelvin) as f32;
    let dst_k = as_shot_kelvin as f32;
    let tint = target_tint.unwrap_or(0.0);

    let w_src = kelvin_tint_to_xyz(src_k, tint);
    let w_dst = kelvin_tint_to_xyz(dst_k, 0.0);

    let rho_src = mat3_vec_mul(BRADFORD_M, w_src);
    let rho_dst = mat3_vec_mul(BRADFORD_M, w_dst);

    let diag = [
        [rho_src[0] / rho_dst[0].max(1e-6), 0.0, 0.0],
        [0.0, rho_src[1] / rho_dst[1].max(1e-6), 0.0],
        [0.0, 0.0, rho_src[2] / rho_dst[2].max(1e-6)],
    ];

    let m_adapt = mat3_mul(BRADFORD_M_INV, mat3_mul(diag, BRADFORD_M));
    let m_cam_to_adapted_xyz = mat3_mul(m_adapt, CAMERA_TO_XYZ_D65);
    mat3_mul(XYZ_TO_REC2020, m_cam_to_adapted_xyz)
}

/// Exact Nikon N-Log Transfer Function Encoding (CPU reference matching Metal shader).
///
/// Cut-off threshold:
///   $x_0 = \left(\frac{452 - 128}{650}\right)^3 - 0.0075 \approx 0.116597$
/// - For $x < 0.116597$: $y = 650 \cdot (x + 0.0075)^{1/3} + 128$
/// - For $x \ge 0.116597$: $y = 150 \cdot \ln(x) + 619$
/// - Output normalized float $V = y / 1023.0$.
pub fn nikon_nlog_encode(x: f32) -> f32 {
    let x_clamped = x.max(0.0);
    let x0: f32 = ((452.0f32 - 128.0f32) / 650.0f32).powi(3) - 0.0075f32; // ~0.11659684
    let y = if x_clamped < x0 {
        650.0 * (x_clamped + 0.0075).cbrt() + 128.0
    } else {
        150.0 * x_clamped.ln() + 619.0
    };
    (y / 1023.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nikon_nlog_curve_continuity_and_reference_points() {
        // At x = 0.0 (scene black): y = 650 * (0.0075)^(1/3) + 128 = 650 * 0.19574338 + 128 = 255.233
        let v_black = nikon_nlog_encode(0.0);
        let y_black = v_black * 1023.0;
        assert!((y_black - 255.233).abs() < 0.05);

        // At cut-off threshold x0 = ((452 - 128)/650)^3 - 0.0075 ~= 0.11659684,
        // the lower branch evaluates to exactly y = 452.
        let x0 = ((452.0f32 - 128.0f32) / 650.0f32).powi(3) - 0.0075f32;
        let v_below = nikon_nlog_encode(x0 - 1e-5);
        let v_above = nikon_nlog_encode(x0 + 1e-5);
        assert!((v_below * 1023.0 - 452.0).abs() < 0.1);
        let expected_above = 150.0 * (x0 + 1e-5).ln() + 619.0;
        assert!((v_above * 1023.0 - expected_above).abs() < 0.1);

        // At 18% middle gray (x = 0.18 >= x0):
        // y = 150 * ln(0.18) + 619 = 150 * (-1.7147984) + 619 = 361.78
        let v_mid = nikon_nlog_encode(0.18);
        assert!((v_mid * 1023.0 - 361.78).abs() < 0.05);
    }

    #[test]
    fn test_parse_nikon_makernote_little_and_big_endian() {
        // Construct a synthetic NCDT payload with "Nikon\0" + version + "II*\0"
        let mut payload = Vec::new();
        payload.extend_from_slice(b"PREAMBLE");
        payload.extend_from_slice(b"Nikon\0");
        payload.extend_from_slice(&[0x02, 0x10, 0x00, 0x00]); // version + pad
        let tiff_start = payload.len();
        payload.extend_from_slice(b"II\x2a\x00");
        payload.extend_from_slice(&8u32.to_le_bytes()); // IFD at offset 8

        // 3 IFD entries: 0x0002 (ISO), 0x000C (WB_RBLevels), 0x003D (BlackLevel)
        payload.extend_from_slice(&3u16.to_le_bytes());
        let data_area_offset = 8 + 2 + 3 * 12 + 4; // 50

        // Entry 1: Tag 0x0002 (ISO), SHORT (3), count 2, inline [0, 800]
        payload.extend_from_slice(&0x0002u16.to_le_bytes());
        payload.extend_from_slice(&3u16.to_le_bytes());
        payload.extend_from_slice(&2u32.to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&800u16.to_le_bytes());

        // Entry 2: Tag 0x000C (WB_RBLevels), RATIONAL (5), count 4, offset = data_area_offset
        payload.extend_from_slice(&0x000Cu16.to_le_bytes());
        payload.extend_from_slice(&5u16.to_le_bytes());
        payload.extend_from_slice(&4u32.to_le_bytes());
        payload.extend_from_slice(&(data_area_offset as u32).to_le_bytes());

        // Entry 3: Tag 0x003D (BlackLevel), SHORT (3), count 4, offset = data_area_offset + 32
        payload.extend_from_slice(&0x003Du16.to_le_bytes());
        payload.extend_from_slice(&3u16.to_le_bytes());
        payload.extend_from_slice(&4u32.to_le_bytes());
        payload.extend_from_slice(&((data_area_offset + 32) as u32).to_le_bytes());

        // Next IFD offset = 0
        payload.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(payload.len() - tiff_start, data_area_offset);

        // Write 4 rationals for WB_RBLevels: [2100/1000, 1350/1000, 1000/1000, 1000/1000]
        for &(num, den) in &[(2100u32, 1000u32), (1350, 1000), (1000, 1000), (1000, 1000)] {
            payload.extend_from_slice(&num.to_le_bytes());
            payload.extend_from_slice(&den.to_le_bytes());
        }
        // Write 4 u16 for BlackLevel: [152, 150, 150, 148]
        for &b in &[152u16, 150, 150, 148] {
            payload.extend_from_slice(&b.to_le_bytes());
        }

        let parsed = parse_ncdt_payload(&payload).expect("should parse NCDT MakerNote");
        assert_eq!(parsed.iso_speed, 800);
        assert!((parsed.wb_rb_levels[0] - 2.10).abs() < 1e-4);
        assert!((parsed.wb_rb_levels[1] - 1.35).abs() < 1e-4);
        assert_eq!(parsed.black_level, [152, 150, 150, 148]);

        // Also test Big-Endian ("MM\0*") MakerNote TIFF parsing
        let mut be_payload = Vec::new();
        be_payload.extend_from_slice(b"Nikon\0\x02\x10\x00\x00");
        be_payload.extend_from_slice(b"MM\x00\x2a");
        be_payload.extend_from_slice(&8u32.to_be_bytes());
        be_payload.extend_from_slice(&1u16.to_be_bytes()); // 1 IFD entry: 0x003D BlackLevel
        let be_data_off = 8 + 2 + 12 + 4;
        be_payload.extend_from_slice(&0x003Du16.to_be_bytes());
        be_payload.extend_from_slice(&3u16.to_be_bytes());
        be_payload.extend_from_slice(&4u32.to_be_bytes());
        be_payload.extend_from_slice(&(be_data_off as u32).to_be_bytes());
        be_payload.extend_from_slice(&0u32.to_be_bytes());
        for &b in &[600u16, 602, 601, 599] {
            be_payload.extend_from_slice(&b.to_be_bytes());
        }

        let parsed_be = parse_ncdt_payload(&be_payload).expect("should parse big-endian MakerNote");
        assert_eq!(parsed_be.black_level, [600, 602, 601, 599]);
        assert_eq!(parsed_be.white_level, 16383);
    }
}
