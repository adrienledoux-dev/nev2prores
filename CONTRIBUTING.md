# Contributing to `nev2prores`

Thank you for contributing to `nev2prores Studio`! Because this project processes high-bit-depth camera RAW video on Apple Silicon Metal GPUs and hardware `VideoToolbox` encoders, we enforce strict automated checks so regressions in color science, QuickTime container compliance, or memory safety never reach `main`.

---

## Development Prerequisites

- **Hardware**: Mac with Apple Silicon (`aarch64-apple-darwin` — M1 / M2 / M3 / M4)
- **OS**: macOS 13.0 Ventura or newer
- **Toolchain**: Rust `1.88+` (`rustup update stable`) and Xcode Command Line Tools (`xcode-select --install`)

---

## Local Pre-PR Verification Checklist

Before opening a Pull Request, run the exact same checks that GitHub Actions CI runs:

```bash
# 1. Format all Rust source files
cargo fmt --all -- --check

# 2. Enforce zero Clippy warnings across library, binaries, and tests
cargo clippy --all-targets -- -D warnings

# 3. Execute unit tests, Metal GPU compute tests, and VideoToolbox round-trip tests
cargo test --all-targets

# 4. Verify macOS .app and .dmg packaging
./scripts/bundle_macos_app.sh
```

---

## Architectural Guardrails

1. **Zero FFmpeg Subprocess Wrappers**: All demuxing, Metal compute debayering, QuickTime `.mov` muxing, and `VideoToolbox` encoding must remain native in-process Rust + macOS frameworks.
2. **Color Science Invariants**:
   - All White Balance (Bradford chromatic adaptation) and Exposure ($\Delta\text{EV}$) math must happen in **scene-linear light** *prior* to Nikon N-Log encoding.
   - Any changes to `shaders/pipeline.metal` or `src/metadata.rs` must keep `test_nikon_nlog_curve_continuity_and_reference_points`, `test_metal_gpu_clipped_sensor_highlights_remain_neutral_white`, and `test_full_end_to_end_nev_to_prores4444_and_fcpxml` passing.
3. **`unsafe` FFI Documentation**: Every `unsafe` block calling `VideoToolbox`, `CoreMedia`, `CoreVideo`, `Metal`, or `dlopen` must carry a `// SAFETY:` comment explaining pointer validity, ownership, and deterministic `CFRelease` cleanup.
4. **No Binary Artifacts in Git**: Do not commit `.mov`, `.nev`, `.app`, or `.dmg` binaries into the repository. Synthetic `.nev` fixtures are generated deterministically in memory during tests via `demuxer::build_synthetic_nev`.

---

## Maintainer Guide: Protecting `main` & Enabling GitHub Pages

To ensure no broken code can be merged into `main` and to publish the download site:

1. **Enable GitHub Pages**:
   - Go to **Settings → Pages** in your GitHub repository.
   - Under **Build and deployment → Source**, choose **GitHub Actions**.
   - `.github/workflows/pages.yml` will automatically deploy `docs/index.html` to `https://adrienledoux-dev.github.io/nev2prores/`.
2. **Create a Branch Ruleset for `main`**:
   - Go to **Settings → Rules → Rulesets → New ruleset → New branch ruleset**.
   - Name it `Protect main`, set **Enforcement status** to **Active**, and target the **Default branch (`main`)**.
   - Enable:
     - **Restrict deletions** & **Block force pushes**
     - **Require a pull request before merging** (1 approval, dismiss stale approvals when new commits are pushed)
     - **Require status checks to pass** → add required check **`Apple Silicon (Metal + VideoToolbox + Clippy)`**
3. **Publishing a New Release (`nev2prores.dmg`)**:
   - Bump `version` in `Cargo.toml`, merge to `main`, then tag and push:
     ```bash
     git tag v0.2.0
     git push origin v0.2.0
     ```
   - `.github/workflows/release.yml` will run all tests on `macos-15`, build & sign `dist/nev2prores.dmg` + `dist/SHA256SUMS.txt`, and publish the GitHub Release so the website's Download button serves it automatically.
   - For Apple Gatekeeper signing & notarization setup, see [`docs/MACOS_SIGNING_AND_NOTARIZATION.md`](docs/MACOS_SIGNING_AND_NOTARIZATION.md).
