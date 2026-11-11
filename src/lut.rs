use std::fs;
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LutError {
    #[error("Failed to read .cube LUT file: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid or missing LUT_3D_SIZE in .cube file (supported range: 2..=129)")]
    InvalidSize,
    #[error("Malformed RGB triplet on line {line}: '{content}'")]
    MalformedLine { line: usize, content: String },
    #[error("Incomplete 3D LUT table: expected {expected} RGB entries, found {actual}")]
    EntryCountMismatch { expected: usize, actual: usize },
}

/// Industry-standard 3D `.cube` Lookup Table (e.g., Nikon Official `N-Log to Rec.709` 33x33x33 LUT)
/// with trilinear interpolation for accurate monitoring in the Studio viewport.
#[derive(Debug, Clone)]
pub struct CubeLut3D {
    pub title: String,
    pub size: usize,
    pub domain_min: [f32; 3],
    pub domain_max: [f32; 3],
    /// Flat array of `size * size * size` RGB entries ordered R-fastest, G-middle, B-slowest.
    pub table: Vec<[f32; 3]>,
}

impl CubeLut3D {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, LutError> {
        let text = fs::read_to_string(path.as_ref())?;
        let fallback_title = path
            .as_ref()
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("Custom 3D LUT")
            .to_string();
        Self::parse_cube_str(&text, &fallback_title)
    }

    pub fn parse_cube_str(content: &str, fallback_title: &str) -> Result<Self, LutError> {
        let mut title = fallback_title.to_string();
        let mut size: Option<usize> = None;
        let mut domain_min = [0.0f32; 3];
        let mut domain_max = [1.0f32; 3];
        let mut table: Vec<[f32; 3]> = Vec::new();

        for (line_idx, raw_line) in content.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            if let Some(rest) = line.strip_prefix("TITLE") {
                let cleaned = rest.trim().trim_matches('"').trim();
                if !cleaned.is_empty() {
                    title = cleaned.to_string();
                }
                continue;
            }

            if let Some(rest) = line.strip_prefix("LUT_3D_SIZE") {
                let parsed: usize = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .ok_or(LutError::InvalidSize)?;
                if !(2..=129).contains(&parsed) {
                    return Err(LutError::InvalidSize);
                }
                size = Some(parsed);
                table.reserve(parsed * parsed * parsed);
                continue;
            }

            if let Some(rest) = line.strip_prefix("DOMAIN_MIN") {
                if let Some(rgb) = parse_rgb_triplet(rest) {
                    domain_min = rgb;
                }
                continue;
            }

            if let Some(rest) = line.strip_prefix("DOMAIN_MAX") {
                if let Some(rgb) = parse_rgb_triplet(rest) {
                    domain_max = rgb;
                }
                continue;
            }

            // Skip optional header keywords like LUT_1D_SIZE or LUT_3D_INPUT_RANGE
            if line.chars().next().is_some_and(|c| c.is_ascii_alphabetic()) {
                continue;
            }

            let rgb = parse_rgb_triplet(line).ok_or_else(|| LutError::MalformedLine {
                line: line_idx + 1,
                content: line.to_string(),
            })?;
            table.push(rgb);
        }

        let n = size.ok_or(LutError::InvalidSize)?;
        let expected = n * n * n;
        if table.len() != expected {
            return Err(LutError::EntryCountMismatch {
                expected,
                actual: table.len(),
            });
        }

        Ok(Self {
            title,
            size: n,
            domain_min,
            domain_max,
            table,
        })
    }

    /// Evaluates the 3D LUT using trilinear interpolation.
    #[inline]
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        let n = self.size;
        let max_idx = (n - 1) as f32;

        let map_coord = |val: f32, c: usize| -> (usize, usize, f32) {
            let span = (self.domain_max[c] - self.domain_min[c]).max(1e-6);
            let norm = ((val - self.domain_min[c]) / span).clamp(0.0, 1.0);
            let pos = norm * max_idx;
            let i0 = (pos.floor() as usize).min(n - 1);
            let i1 = (i0 + 1).min(n - 1);
            let frac = pos - (i0 as f32);
            (i0, i1, frac)
        };

        let (r0, r1, fr) = map_coord(rgb[0], 0);
        let (g0, g1, fg) = map_coord(rgb[1], 1);
        let (b0, b1, fb) = map_coord(rgb[2], 2);

        let idx = |r: usize, g: usize, b: usize| -> [f32; 3] { self.table[(b * n + g) * n + r] };

        let c000 = idx(r0, g0, b0);
        let c100 = idx(r1, g0, b0);
        let c010 = idx(r0, g1, b0);
        let c110 = idx(r1, g1, b0);
        let c001 = idx(r0, g0, b1);
        let c101 = idx(r1, g0, b1);
        let c011 = idx(r0, g1, b1);
        let c111 = idx(r1, g1, b1);

        let lerp = |a: [f32; 3], b: [f32; 3], t: f32| -> [f32; 3] {
            [
                a[0] + (b[0] - a[0]) * t,
                a[1] + (b[1] - a[1]) * t,
                a[2] + (b[2] - a[2]) * t,
            ]
        };

        let c00 = lerp(c000, c100, fr);
        let c10 = lerp(c010, c110, fr);
        let c01 = lerp(c001, c101, fr);
        let c11 = lerp(c011, c111, fr);

        let c0 = lerp(c00, c10, fg);
        let c1 = lerp(c01, c11, fg);

        lerp(c0, c1, fb)
    }
}

fn parse_rgb_triplet(s: &str) -> Option<[f32; 3]> {
    let mut parts = s.split_whitespace();
    let r: f32 = parts.next()?.parse().ok()?;
    let g: f32 = parts.next()?.parse().ok()?;
    let b: f32 = parts.next()?.parse().ok()?;
    if r.is_finite() && g.is_finite() && b.is_finite() {
        Some([r, g, b])
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cube_lut_3d_parse_and_trilinear_identity() {
        let mut cube =
            String::from("# Comment header\nTITLE \"Nikon N-Log Identity Test\"\nLUT_3D_SIZE 4\n");
        for b in 0..4 {
            for g in 0..4 {
                for r in 0..4 {
                    cube.push_str(&format!(
                        "{:.6} {:.6} {:.6}\n",
                        r as f32 / 3.0,
                        g as f32 / 3.0,
                        b as f32 / 3.0
                    ));
                }
            }
        }

        let lut = CubeLut3D::parse_cube_str(&cube, "fallback").expect("valid 4x4x4 cube");
        assert_eq!(lut.title, "Nikon N-Log Identity Test");
        assert_eq!(lut.size, 4);

        let sample = [0.18f32, 0.42f32, 0.87f32];
        let out = lut.apply(sample);
        assert!((out[0] - sample[0]).abs() < 1e-5);
        assert!((out[1] - sample[1]).abs() < 1e-5);
        assert!((out[2] - sample[2]).abs() < 1e-5);
    }
}
