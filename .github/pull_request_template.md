## Summary

<!-- Describe what this PR changes and why. Link any related issue (e.g. Fixes #12). -->

## Pipeline Area Affected

- [ ] Stage 1 — 64-Bit ISO-BMFF Demuxer / Nikon `NCDT` Metadata (`src/demuxer.rs`, `src/metadata.rs`)
- [ ] Stage 2 — TicoRAW Decoder / Dynamic Plugin Loader (`src/decoder.rs`)
- [ ] Stage 3 — Apple Metal Compute Debayer & N-Log Color Science (`shaders/pipeline.metal`, `src/metal_pipeline.rs`)
- [ ] Stage 4 — `VideoToolbox` ProRes Encoder / `.mov` Muxer / FCPXML (`src/encoder.rs`, `src/fcpxml.rs`)
- [ ] Studio GUI / 3D `.cube` LUTs / Scopes (`src/gui.rs`, `src/lut.rs`)
- [ ] Packaging / CI / Documentation (`scripts/`, `.github/`, `docs/`)

## Verification Checklist

- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo clippy --all-targets -- -D warnings` passes with zero warnings
- [ ] `cargo test --all-targets` passes (including Metal GPU & `VTDecompressionSession` ProRes round-trip tests)
- [ ] Any new `unsafe` FFI blocks include a `// SAFETY:` invariant comment
