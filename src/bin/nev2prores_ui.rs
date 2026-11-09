use std::path::PathBuf;

use anyhow::Result;
use nev2prores::gui::run_native_gui;

fn main() -> Result<()> {
    // Support opening a .nev file or folder passed by macOS LaunchServices / Finder
    // (ignoring macOS `-psn_0_...` Process Serial Number arguments).
    let initial_input = std::env::args()
        .skip(1)
        .filter(|arg| !arg.starts_with("-psn_") && !arg.starts_with('-'))
        .map(PathBuf::from)
        .find(|p| p.exists());

    run_native_gui(initial_input, None)
}
