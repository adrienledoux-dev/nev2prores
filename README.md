<p align="center">
  <img src="assets/logo_1024.png" width="180" alt="nev2prores Studio Logo"/>
</p>

<h1 align="center">nev2prores Studio</h1>

<p align="center">
  <b>Native macOS Apple Silicon Nikon N-RAW (<code>.nev</code>) to ProRes 4444 XQ / 4444, Lossless 16-Bit RGB &amp; N-Log Studio</b><br/>
  <sub>Compatible with all Apple Silicon Macs (M-series — M1 through M5 / M6 &amp; newer) running macOS 13 Ventura or later</sub>
</p>

<p align="center">
  <a href="https://github.com/adrienledoux-dev/nev2prores/releases/latest/download/nev2prores.dmg">
    <img src="https://img.shields.io/badge/%E2%AC%87_Download_nev2prores.dmg-macOS_Apple_Silicon-f5c438?style=for-the-badge&logo=apple&logoColor=black" alt="Download nev2prores.dmg for macOS Apple Silicon"/>
  </a>
</p>

<p align="center">
  <a href="https://github.com/adrienledoux-dev/nev2prores/actions/workflows/ci.yml"><img src="https://github.com/adrienledoux-dev/nev2prores/actions/workflows/ci.yml/badge.svg" alt="CI Status"/></a>
  <a href="https://github.com/adrienledoux-dev/nev2prores/releases/latest"><img src="https://img.shields.io/github/v/release/adrienledoux-dev/nev2prores?label=Latest%20Release" alt="Latest Release"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache-2.0"/></a>
</p>

<p align="center">
  <a href="https://adrienledoux-dev.github.io/nev2prores/"><b>🌐 Website &amp; Download</b></a> •
  <a href="https://github.com/adrienledoux-dev/nev2prores/releases/latest/download/nev2prores.dmg"><b>💿 Direct .DMG Installer</b></a> •
  <a href="https://github.com/adrienledoux-dev/nev2prores/issues/new?template=feature_request.yml"><b>💡 Suggest an Improvement</b></a> •
  <a href="docs/MACOS_SIGNING_AND_NOTARIZATION.md"><b>🔐 Signing Guide</b></a> •
  <a href="CONTRIBUTING.md"><b>🤝 Contributing</b></a>
</p>

---

## Overview

`nev2prores` is an open-source, native macOS (`aarch64-apple-darwin`) CLI and Studio GUI built in **Rust + Apple Metal Compute Shaders + Apple `VideoToolbox`**. It decodes **12-bit and 14-bit** `.nev` (N-RAW / TicoRAW) containers and transcodes them in **32-bit float scene-linear light** into:

- **Apple ProRes 4444 XQ (`ap4x`, 12-bit 4:4:4 Extreme Quality)** — Apple's highest-bitrate cinema profile designed for 14-bit+ dynamic range log camera sensors
- **Apple ProRes 4444 (`ap4h`, 12-bit 4:4:4)** — unsubsampled 12-bit N-Log mastering
- **Apple ProRes 422 HQ (`apch`, 10-bit 4:2:2)** — lightweight broadcast & editorial mastering
- **Uncompressed 16-Bit RGBA (`b64a`, 16-bit 4:4:4:4 Lossless)** — preserves full $\ge 14\text{-bit}$ (`16,384`+) / `16-bit` (`65,536`) code values with zero DCT compression

Zero FFmpeg subprocess wrappers are used.

---

## Key Features (v0.2.1)

- **12-Bit & 14-Bit Scene-Linear RAW Color Science (`shaders/pipeline.metal`, `src/decoder.rs`, `src/metadata.rs`)**
  - Parallel 12-bit (`3B -> 2px`), 14-bit (`7B -> 4px`), 16-bit LE, and LeGall 5/3 wavelet Bayer unpacking (`WhiteLevel = 4095` for 12-bit, `16383` for 14-bit).
  - 5×5 **Malvar-He-Cutler** edge-directed RGGB demosaicing in 32-bit IEEE float (`Full` 1:1 native or `Half` 2×2 superpixel proxy).
  - **Neutral Highlight Reconstruction** prior to chromatic adaptation so clipped sensor channels never turn magenta when warming/cooling Kelvin.
  - $3 \times 3$ **Bradford Chromatic Adaptation** ($M_{\text{WB}}$) into ITU-R BT.2020 + mired-interpolated As-Shot Kelvin estimation from `WB_RBLevels` (`0x000C`).
  - Official continuous **Nikon N-Log** transfer curve ($0\%$ black at CV `128`, $18\%$ gray at CV `372`, $C^0/C^1$ splice at $x_0 \approx 0.328189$).
- **Interactive macOS Studio GUI (`nev2prores-ui`)**
  - Real-time 60+ fps Apple Metal viewport with Dark/Light themes and Pros & Cons tooltips on every setting.
  - **Per-Clip RAW Grading** (`★` indicator) + one-click **Copy Current Grade to All Clips**.
  - **Custom 3D `.cube` LUT Loader** (17³ / 33³ / 65³ trilinear interpolation for non-destructive show-LUT or official Nikon LUT monitoring).
  - **Live 64-Bin RGB N-Log Histogram Scope** (with `BLK 128` & `18% 372` reference lines), **Highlight Clipping Zebras**, and hover pixel inspector.
  - **Cancellable Background Transcoding** that cleanly removes partial `.mov` files if stopped mid-export.
- **Broadcast-Compliant QuickTime `.mov` & Final Cut Pro FCPXML v1.11 (`src/encoder.rs`, `src/fcpxml.rs`)**
  - IOSurface-backed `VTCompressionSessionGetPixelBufferPool` hardware encoding (`ap4x`, `ap4h`, `apch`) + lossless 16-bit `b64a`.
  - 64-bit `mdat` / `co64` + automatic v1 64-bit `mvhd`/`tkhd`/`mdhd` headers, `colr` (`nclc` `9-1-9`), QuickTime v1 `in24` 24-bit little-endian PCM audio (`wave`/`enda`), and `tmcd` camera timecode passthrough.
  - Optional dynamic `dlopen` C ABI plugin bridge (`NEV2PRORES_TICORAW_DYLIB`) for external commercial intoPIX TicoRAW decoders.

---

## Installation (macOS 13+ — All Apple Silicon M-Series Macs)

### Option A: One-Click Drag-and-Drop `.dmg`
1. Click **[⬇ Download `nev2prores.dmg`](https://github.com/adrienledoux-dev/nev2prores/releases/latest/download/nev2prores.dmg)**.
2. Open `nev2prores.dmg` in Finder and drag **`nev2prores.app`** onto **`Applications`**.
3. *(If running an unsigned community build prior to Apple Developer ID notarization, clear macOS quarantine once via `xattr -dr com.apple.quarantine /Applications/nev2prores.app`.)*

### Option B: Build & Bundle Locally from Source
```bash
git clone https://github.com/adrienledoux-dev/nev2prores.git
cd nev2prores

# Run unit, 14-bit RAW, Metal GPU, and VideoToolbox ProRes round-trip tests
cargo test --all-targets

# Build dist/nev2prores.app, dist/nev2prores.dmg, and dist/SHA256SUMS.txt
./scripts/bundle_macos_app.sh
cp -R dist/nev2prores.app /Applications/
```

---

## CLI Usage

```bash
nev2prores \
  --input <FILE_OR_DIR> \
  --output <DIR> \
  --codec <prores4444xq|prores4444|prores422hq|raw16> \
  --kelvin <OPTIONAL_TARGET_KELVIN> \
  --tint <OPTIONAL_TINT_SHIFT> \
  --ev <OPTIONAL_EXPOSURE_STOPS_FLOAT> \
  --scale <full|half> \
  --fcpxml \
  --inspect-only
```

### Generate a Synthetic 14-Bit RGGB `.nev` Sample & Transcode to ProRes 4444 XQ
```bash
cargo run --release --bin nev2prores -- \
  --generate-sample \
  --sample-bit-depth 14 \
  --input ./sample_14bit.nev \
  --output ./out \
  --codec prores4444xq \
  --kelvin 5200 \
  --tint 2.0 \
  --ev 0.5 \
  --fcpxml
```

---

## Contributing & Feedback

Contributions, camera metadata test reports, and feature ideas are welcome!
- [Suggest an Improvement](https://github.com/adrienledoux-dev/nev2prores/issues/new?template=feature_request.yml) or [Report a Bug](https://github.com/adrienledoux-dev/nev2prores/issues/new?template=bug_report.yml)
- Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for our Apple Silicon CI checks and branch-protection setup
- Read [`docs/MACOS_SIGNING_AND_NOTARIZATION.md`](docs/MACOS_SIGNING_AND_NOTARIZATION.md) to configure Apple Developer ID signing and notarization

---

## License & Trademark Disclaimer

Licensed under the [Apache License, Version 2.0](LICENSE).

`nev2prores` is an independent open-source software project and is **not** affiliated with, endorsed by, or sponsored by Nikon Corporation, intoPIX SA, or Apple Inc. Nikon, N-RAW, and N-Log are trademarks or registered trademarks of Nikon Corporation. TicoRAW is a trademark or registered trademark of intoPIX SA. Apple, macOS, Final Cut Pro, Metal, VideoToolbox, QuickTime, and ProRes are trademarks of Apple Inc.
