#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

APP_NAME="nev2prores"
BUNDLE_DIR="$REPO_ROOT/dist/${APP_NAME}.app"
DMG_STAGING="$REPO_ROOT/dist/dmg_staging"
DMG_PATH="$REPO_ROOT/dist/${APP_NAME}.dmg"
ASSETS_DIR="$REPO_ROOT/assets"

mkdir -p "$ASSETS_DIR"

echo "[1/4] Preparing high-resolution macOS squircle icon (assets/logo_1024.png)..."
if [ -f "$ASSETS_DIR/app_icon_source.jpg" ]; then
    sips -s format png -z 1024 1024 "$ASSETS_DIR/app_icon_source.jpg" --out "$ASSETS_DIR/logo_raw_1024.png" >/dev/null
    python3 - "$ASSETS_DIR/logo_raw_1024.png" "$ASSETS_DIR/logo_1024.png" << 'PYEOF'
import struct, sys, zlib, math

src_path, dst_path = sys.argv[1], sys.argv[2]

with open(src_path, "rb") as f:
    data = f.read()

assert data[:8] == b"\x89PNG\r\n\x1a\n"
pos = 8
width = height = bit_depth = color_type = 0
idat_chunks = []
while pos < len(data):
    length = struct.unpack(">I", data[pos:pos+4])[0]
    chunk_type = data[pos+4:pos+8]
    chunk_data = data[pos+8:pos+8+length]
    pos += 12 + length
    if chunk_type == b"IHDR":
        width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk_data)
    elif chunk_type == b"IDAT":
        idat_chunks.append(chunk_data)

raw = zlib.decompress(b"".join(idat_chunks))
bpp = 4 if color_type == 6 else 3
stride = width * bpp

# Decode PNG scanlines (supporting filter types 0..4)
recon = bytearray(height * stride)
in_pos = 0
for y in range(height):
    ftype = raw[in_pos]
    in_pos += 1
    row_start = y * stride
    for x in range(stride):
        val = raw[in_pos]
        in_pos += 1
        a = recon[row_start + x - bpp] if x >= bpp else 0
        b = recon[row_start - stride + x] if y > 0 else 0
        c = recon[row_start - stride + x - bpp] if (y > 0 and x >= bpp) else 0
        if ftype == 1:
            val = (val + a) & 0xFF
        elif ftype == 2:
            val = (val + b) & 0xFF
        elif ftype == 3:
            val = (val + ((a + b) >> 1)) & 0xFF
        elif ftype == 4:
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
            val = (val + pr) & 0xFF
        recon[row_start + x] = val

# Crop/scale the inner squircle region [118..906] to standard macOS icon bounds [100..924]
# and apply a smooth anti-aliased superellipse (squircle) alpha mask outside the icon body.
out_rows = []
cx = cy = 511.5
half_box = 412.0  # standard macOS 824x824 icon body inside 1024x1024 canvas
corner_r = 185.0  # Apple continuous corner radius

src_min = 122.0
src_span = 1024.0 - 2.0 * src_min

for y in range(height):
    row = bytearray([0])
    dy_box = max(abs(y - cy) - (half_box - corner_r), 0.0)
    for x in range(width):
        dx_box = max(abs(x - cx) - (half_box - corner_r), 0.0)
        dist = math.hypot(dx_box, dy_box)
        if dist > corner_r + 1.0:
            row.extend((0, 0, 0, 0))
            continue

        alpha = 255
        if dist > corner_r - 1.5:
            alpha = int(max(0.0, min(1.0, (corner_r + 0.5 - dist) / 2.0)) * 255.0)

        # Map output (x, y) in [100..924] to source artwork coordinates
        u = (x - (cx - half_box)) / (2.0 * half_box)
        v = (y - (cy - half_box)) / (2.0 * half_box)
        sx = int(max(0, min(width - 1, round(src_min + u * src_span))))
        sy = int(max(0, min(height - 1, round(src_min + v * src_span))))
        idx = sy * stride + sx * bpp
        r, g, b = recon[idx], recon[idx + 1], recon[idx + 2]
        row.extend((r, g, b, alpha))
    out_rows.append(bytes(row))

def make_chunk(tag, payload):
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
png_bytes = (
    b"\x89PNG\r\n\x1a\n"
    + make_chunk(b"IHDR", ihdr)
    + make_chunk(b"IDAT", zlib.compress(b"".join(out_rows), 9))
    + make_chunk(b"IEND", b"")
)
with open(dst_path, "wb") as f:
    f.write(png_bytes)
PYEOF
    rm -f "$ASSETS_DIR/logo_raw_1024.png"
    sips -z 256 256 "$ASSETS_DIR/logo_1024.png" --out "$ASSETS_DIR/logo_256.png" >/dev/null
fi

echo "[2/4] Building optimized Apple Silicon release binaries (nev2prores-ui & nev2prores)..."
cargo build --release --bins

echo "[3/4] Assembling macOS application bundle at: $BUNDLE_DIR"
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/Contents/MacOS"
mkdir -p "$BUNDLE_DIR/Contents/Resources"

cp "target/release/nev2prores-ui" "$BUNDLE_DIR/Contents/MacOS/nev2prores-ui"
cp "target/release/nev2prores" "$BUNDLE_DIR/Contents/MacOS/nev2prores"
chmod +x "$BUNDLE_DIR/Contents/MacOS/nev2prores-ui" "$BUNDLE_DIR/Contents/MacOS/nev2prores"

printf "APPL????" > "$BUNDLE_DIR/Contents/PkgInfo"

cat > "$BUNDLE_DIR/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>nev2prores</string>
    <key>CFBundleDisplayName</key>
    <string>nev2prores Studio</string>
    <key>CFBundleIdentifier</key>
    <string>com.nikonraw.nev2prores</string>
    <key>CFBundleVersion</key>
    <string>0.1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleExecutable</key>
    <string>nev2prores-ui</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSSupportsAutomaticGraphicsSwitching</key>
    <true/>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeName</key>
            <string>Nikon N-RAW Video (.nev)</string>
            <key>CFBundleTypeRole</key>
            <string>Viewer</string>
            <key>LSHandlerRank</key>
            <string>Alternate</string>
            <key>CFBundleTypeExtensions</key>
            <array>
                <string>nev</string>
                <string>NEV</string>
            </array>
        </dict>
    </array>
</dict>
</plist>
EOF

ICONSET_DIR="$REPO_ROOT/dist/AppIcon.iconset"
rm -rf "$ICONSET_DIR"
mkdir -p "$ICONSET_DIR"

sips -z 16 16     "$ASSETS_DIR/logo_1024.png" --out "$ICONSET_DIR/icon_16x16.png"      >/dev/null
sips -z 32 32     "$ASSETS_DIR/logo_1024.png" --out "$ICONSET_DIR/icon_16x16@2x.png"   >/dev/null
sips -z 32 32     "$ASSETS_DIR/logo_1024.png" --out "$ICONSET_DIR/icon_32x32.png"      >/dev/null
sips -z 64 64     "$ASSETS_DIR/logo_1024.png" --out "$ICONSET_DIR/icon_32x32@2x.png"   >/dev/null
sips -z 128 128   "$ASSETS_DIR/logo_1024.png" --out "$ICONSET_DIR/icon_128x128.png"    >/dev/null
sips -z 256 256   "$ASSETS_DIR/logo_1024.png" --out "$ICONSET_DIR/icon_128x128@2x.png" >/dev/null
sips -z 256 256   "$ASSETS_DIR/logo_1024.png" --out "$ICONSET_DIR/icon_256x256.png"    >/dev/null
sips -z 512 512   "$ASSETS_DIR/logo_1024.png" --out "$ICONSET_DIR/icon_256x256@2x.png" >/dev/null
sips -z 512 512   "$ASSETS_DIR/logo_1024.png" --out "$ICONSET_DIR/icon_512x512.png"    >/dev/null
cp "$ASSETS_DIR/logo_1024.png" "$ICONSET_DIR/icon_512x512@2x.png"

iconutil -c icns "$ICONSET_DIR" -o "$BUNDLE_DIR/Contents/Resources/AppIcon.icns"
rm -rf "$ICONSET_DIR"

# Also set the custom icon on the .app bundle itself so Finder displays it immediately
touch "$BUNDLE_DIR"
codesign --force --deep --sign - "$BUNDLE_DIR"

echo "[4/4] Creating drag-and-drop macOS Disk Image: $DMG_PATH"
hdiutil detach "/Volumes/nev2prores Studio" >/dev/null 2>&1 || true
rm -rf "$DMG_STAGING" "$DMG_PATH"
mkdir -p "$DMG_STAGING"
cp -R "$BUNDLE_DIR" "$DMG_STAGING/"
ln -s /Applications "$DMG_STAGING/Applications"
hdiutil create -volname "nev2prores Studio" -srcfolder "$DMG_STAGING" -ov -format UDZO "$DMG_PATH" >/dev/null 2>&1
rm -rf "$DMG_STAGING"

echo "================================================================"
echo "✅ Built macOS Application Bundle : $BUNDLE_DIR"
echo "✅ Built Drag-and-Drop DMG Image  : $DMG_PATH"
echo "================================================================"
