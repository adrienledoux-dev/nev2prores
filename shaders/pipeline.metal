#include <metal_stdlib>
using namespace metal;

/// Scene-Linear RAW to Nikon N-Log / Rec. 2020 GPU Pipeline Uniforms.
/// Matches `ColorPipelineUniforms` in `src/metal_pipeline.rs` (`#[repr(C)]`).
struct ColorPipelineUniforms {
    uint width;
    uint height;
    uint out_width;
    uint out_height;
    uint out_stride_elements; // Row stride in `half4` or `ushort4` pixels (bytes_per_row / 8)
    uint scale_factor;        // 1 for full scale, 2 for half scale
    float white_level;        // 4095.0 (12-bit) or 16383.0 (14-bit)
    float ev_scale;           // 2^(delta_EV)
    float4 black_level;       // [R, G1, G2, B] sensor black levels in ADU
    float4 wb_gains;          // [R_gain, G1_gain, G2_gain, B_gain] As-Shot WB multipliers
    float4 color_matrix_row0; // Row 0 of M_WB (Camera -> XYZ -> Bradford -> Rec.2020)
    float4 color_matrix_row1; // Row 1 of M_WB
    float4 color_matrix_row2; // Row 2 of M_WB
};

/// Reflects an out-of-bounds coordinate by multiples of 2 so that the RGGB
/// Bayer parity `(x & 1, y & 1)` is strictly preserved at image boundaries.
inline int mirror_bayer_coord(int coord, int limit) {
    if (coord < 0) {
        int steps = (-coord + 1) / 2;
        coord += steps * 2;
    } else if (coord >= limit) {
        int steps = (coord - limit + 2) / 2;
        coord -= steps * 2;
    }
    return clamp(coord, 0, limit - 1);
}

/// Reads a single RGGB Bayer sample at `(x, y)`, subtracts the channel's sensor
/// black level, normalizes to scene-linear float (`[0.0, 1.0+]`), and applies
/// the channel's As-Shot White Balance gain.
///
/// RGGB layout:
///   (even x, even y) -> R  (channel 0)
///   (odd  x, even y) -> G1 (channel 1)
///   (even x, odd  y) -> G2 (channel 2)
///   (odd  x, odd  y) -> B  (channel 3)
inline float sample_bayer_linear(
    device const ushort* bayer,
    int x,
    int y,
    constant ColorPipelineUniforms& u
) {
    int cx = mirror_bayer_coord(x, int(u.width));
    int cy = mirror_bayer_coord(y, int(u.height));

    uint raw_adu = uint(bayer[cy * int(u.width) + cx]);
    int phase = ((cy & 1) << 1) | (cx & 1);

    float black = u.black_level[phase];
    float denom = max(u.white_level - black, 1.0f);
    float norm = max((float(raw_adu) - black) / denom, 0.0f);

    return norm * u.wb_gains[phase];
}

/// Malvar-He-Cutler (2004) 5x5 edge-directed gradient-corrected RGGB demosaicing.
/// Reconstructs full scene-linear RGB at Bayer coordinate `(x, y)`.
inline float3 malvar_he_cutler_5x5(
    device const ushort* bayer,
    int x,
    int y,
    constant ColorPipelineUniforms& u
) {
    #define S(dx, dy) sample_bayer_linear(bayer, x + (dx), y + (dy), u)

    float c00 = S(0, 0);

    // Immediate cardinal neighbors (distance 1)
    float n_w1 = S(-1, 0);
    float n_e1 = S( 1, 0);
    float n_n1 = S( 0, -1);
    float n_s1 = S( 0,  1);

    // Diagonal neighbors (distance sqrt(2))
    float n_nw = S(-1, -1);
    float n_ne = S( 1, -1);
    float n_sw = S(-1,  1);
    float n_se = S( 1,  1);

    // Cardinal neighbors at distance 2 (same Bayer color as center c00)
    float n_w2 = S(-2, 0);
    float n_e2 = S( 2, 0);
    float n_n2 = S( 0, -2);
    float n_s2 = S( 0,  2);

    #undef S

    int px = x & 1;
    int py = y & 1;

    float r, g, b;

    if (py == 0 && px == 0) {
        // Center is Red (R row, R column)
        r = c00;
        g = (4.0f * c00 + 2.0f * (n_w1 + n_e1 + n_n1 + n_s1) - (n_w2 + n_e2 + n_n2 + n_s2)) * 0.125f;
        b = (6.0f * c00 + 2.0f * (n_nw + n_ne + n_sw + n_se) - 1.5f * (n_w2 + n_e2 + n_n2 + n_s2)) * 0.125f;
    } else if (py == 0 && px == 1) {
        // Center is Green in Red row (G1: horizontal neighbors are R, vertical neighbors are B)
        g = c00;
        r = (5.0f * c00 + 4.0f * (n_w1 + n_e1)
             - (n_nw + n_ne + n_sw + n_se + n_w2 + n_e2)
             + 0.5f * (n_n2 + n_s2)) * 0.125f;
        b = (5.0f * c00 + 4.0f * (n_n1 + n_s1)
             - (n_nw + n_ne + n_sw + n_se + n_n2 + n_s2)
             + 0.5f * (n_w2 + n_e2)) * 0.125f;
    } else if (py == 1 && px == 0) {
        // Center is Green in Blue row (G2: vertical neighbors are R, horizontal neighbors are B)
        g = c00;
        r = (5.0f * c00 + 4.0f * (n_n1 + n_s1)
             - (n_nw + n_ne + n_sw + n_se + n_n2 + n_s2)
             + 0.5f * (n_w2 + n_e2)) * 0.125f;
        b = (5.0f * c00 + 4.0f * (n_w1 + n_e1)
             - (n_nw + n_ne + n_sw + n_se + n_w2 + n_e2)
             + 0.5f * (n_n2 + n_s2)) * 0.125f;
    } else {
        // Center is Blue (B row, B column)
        b = c00;
        g = (4.0f * c00 + 2.0f * (n_w1 + n_e1 + n_n1 + n_s1) - (n_w2 + n_e2 + n_n2 + n_s2)) * 0.125f;
        r = (6.0f * c00 + 2.0f * (n_nw + n_ne + n_sw + n_se) - 1.5f * (n_w2 + n_e2 + n_n2 + n_s2)) * 0.125f;
    }

    return max(float3(r, g, b), float3(0.0f));
}

/// Official Nikon N-Log Transfer Function Encoding.
/// Maps scene-linear reflection `x` into the 10-bit reference domain `y in [0, 1023]`,
/// then normalizes to `V = y / 1023.0` in `[0.0, 1.0]`.
///
/// Cut-off threshold:
///   x_0 = ((452 - 128) / 650)^3 - 0.0075 = (324 / 650)^3 - 0.0075 ~= 0.11659684f
inline float nikon_nlog_encode_scalar(float x) {
    const float x_clamped = max(x, 0.0f);
    const float x0 = 0.11659684f; // ((452.0 - 128.0) / 650.0)^3 - 0.0075

    float y;
    if (x_clamped < x0) {
        y = 650.0f * pow(x_clamped + 0.0075f, 1.0f / 3.0f) + 128.0f;
    } else {
        y = 150.0f * log(x_clamped) + 619.0f;
    }

    return clamp(y / 1023.0f, 0.0f, 1.0f);
}

inline float3 nikon_nlog_encode(float3 rgb_linear) {
    return float3(
        nikon_nlog_encode_scalar(rgb_linear.x),
        nikon_nlog_encode_scalar(rgb_linear.y),
        nikon_nlog_encode_scalar(rgb_linear.z)
    );
}

/// Evaluates the full RAW -> Demosaic -> Exposure -> Bradford Chromatic Adaptation -> Rec.2020 N-Log pipeline
/// for output pixel `gid`.
inline float3 evaluate_pipeline_pixel(
    device const ushort* bayer_in,
    uint2 gid,
    constant ColorPipelineUniforms& u
) {
    float3 cam_linear;
    if (u.scale_factor == 2) {
        // Half-resolution mode: evaluate 2x2 Malvar-He-Cutler superpixel box filter
        int bx = int(gid.x * 2);
        int by = int(gid.y * 2);
        float3 p00 = malvar_he_cutler_5x5(bayer_in, bx,     by,     u);
        float3 p10 = malvar_he_cutler_5x5(bayer_in, bx + 1, by,     u);
        float3 p01 = malvar_he_cutler_5x5(bayer_in, bx,     by + 1, u);
        float3 p11 = malvar_he_cutler_5x5(bayer_in, bx + 1, by + 1, u);
        cam_linear = 0.25f * (p00 + p10 + p01 + p11);
    } else {
        cam_linear = malvar_he_cutler_5x5(bayer_in, int(gid.x), int(gid.y), u);
    }

    // Apply scene-linear EV exposure gain (2^Delta_EV)
    float3 exposed = cam_linear * u.ev_scale;

    // Apply 3x3 Bradford Chromatic Adaptation & Camera -> Rec. 2020 matrix (M_WB)
    float3 rec2020_linear = float3(
        dot(u.color_matrix_row0.xyz, exposed),
        dot(u.color_matrix_row1.xyz, exposed),
        dot(u.color_matrix_row2.xyz, exposed)
    );
    rec2020_linear = max(rec2020_linear, float3(0.0f));

    // Encode to Nikon N-Log [0.0, 1.0]
    return nikon_nlog_encode(rec2020_linear);
}

/// Primary compute kernel writing `kCVPixelFormatType_64RGBAHalf` (`half4` RGBA).
kernel void debayer_nlog_rgba16f(
    device const ushort* bayer_in [[buffer(0)]],
    device half4* rgba_out [[buffer(1)]],
    constant ColorPipelineUniforms& u [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= u.out_width || gid.y >= u.out_height) {
        return;
    }

    float3 nlog_rgb = evaluate_pipeline_pixel(bayer_in, gid, u);

    uint out_idx = gid.y * u.out_stride_elements + gid.x;
    rgba_out[out_idx] = half4(half(nlog_rgb.x), half(nlog_rgb.y), half(nlog_rgb.z), half(1.0f));
}

/// Alternate compute kernel writing `kCVPixelFormatType_4444AYpCbCr16` (`ushort4` AY'CbCr 16-bit, ITU-R BT.2020).
kernel void debayer_nlog_ayuv16(
    device const ushort* bayer_in [[buffer(0)]],
    device ushort4* ayuv_out [[buffer(1)]],
    constant ColorPipelineUniforms& u [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    if (gid.x >= u.out_width || gid.y >= u.out_height) {
        return;
    }

    float3 nlog_rgb = evaluate_pipeline_pixel(bayer_in, gid, u);

    // ITU-R BT.2020 non-constant luminance luma/chroma coefficients
    float kr = 0.2627f;
    float kb = 0.0593f;
    float kg = 1.0f - kr - kb; // 0.6780f

    float y_norm = kr * nlog_rgb.x + kg * nlog_rgb.y + kb * nlog_rgb.z;
    float cb_norm = 0.5f + (nlog_rgb.z - y_norm) / (2.0f * (1.0f - kb));
    float cr_norm = 0.5f + (nlog_rgb.x - y_norm) / (2.0f * (1.0f - kr));

    // Video range 16-bit quantization (Rec. 2020 10/12-bit shifted to 16-bit container)
    ushort a16  = 65535u;
    ushort y16  = ushort(clamp(round((16.0f  + 219.0f * y_norm)  * 256.0f), 0.0f, 65535.0f));
    ushort cb16 = ushort(clamp(round((16.0f  + 224.0f * cb_norm) * 256.0f), 0.0f, 65535.0f));
    ushort cr16 = ushort(clamp(round((16.0f  + 224.0f * cr_norm) * 256.0f), 0.0f, 65535.0f));

    uint out_idx = gid.y * u.out_stride_elements + gid.x;
    ayuv_out[out_idx] = ushort4(a16, y16, cb16, cr16);
}
