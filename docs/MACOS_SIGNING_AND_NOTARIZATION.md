# macOS Code Signing & Apple Notarization Guide (`nev2prores Studio`)

By default, `./scripts/bundle_macos_app.sh` builds an **ad-hoc signed** `.app` and `.dmg`. Ad-hoc signing works out-of-the-box on the Mac that built it, **but** when another user downloads `nev2prores.dmg` from GitHub Releases or your website, macOS Gatekeeper attaches the `com.apple.quarantine` attribute and displays:

> *"nev2prores.app Not Opened — Apple could not verify nev2prores.app is free of malware…"*

To let anyone on macOS 13+ download `nev2prores.dmg`, drag `nev2prores.app` into `/Applications`, and double-click it **with zero Gatekeeper warnings**, you need two things from Apple:
1. **Developer ID Application** signature with the **Hardened Runtime** (`--options runtime --timestamp`), and
2. **Apple Notarization + Ticket Stapling** (`xcrun notarytool` + `xcrun stapler`).

Both [`scripts/bundle_macos_app.sh`](../scripts/bundle_macos_app.sh) and [`.github/workflows/release.yml`](../.github/workflows/release.yml) are **already wired up** to perform this automatically as soon as you supply your credentials. Follow the step-by-step instructions below.

---

## Step 1: Enroll in the Apple Developer Program

1. Sign in with your Apple Account at <https://developer.apple.com/programs/enroll/> (requires an active Apple Developer Program membership, \$99/year).
2. Once approved, note your **10-character Team ID** shown in the top-right corner of <https://developer.apple.com/account> (for example, `A1B2C3D4E5`).

---

## Step 2: Create a `Developer ID Application` Certificate

> **Important:** Choose **Developer ID Application** (used for distributing `.app` and `.dmg` outside the Mac App Store), *not* "Apple Development" or "Mac App Distribution".

### Option A — Via Xcode (Fastest)
1. Open **Xcode → Settings (`⌘,`) → Accounts**.
2. Select your Apple ID on the left, select your Team, and click **Manage Certificates…**.
3. Click the **`+`** button in the bottom-left corner and choose **Developer ID Application**.
4. Verify that the certificate is now in your login Keychain by running in Terminal:
   ```bash
   security find-identity -v -p codesigning
   ```
   Copy the exact quoted string shown next to your certificate — it looks like:
   ```text
   "Developer ID Application: Adrien Ledoux (A1B2C3D4E5)"
   ```

### Option B — Via Apple Developer Portal (Without Full Xcode GUI)
1. Open **Keychain Access** on your Mac → menu bar **Keychain Access → Certificate Assistant → Request a Certificate From a Certificate Authority…**.
2. Enter your email address, select **Saved to disk**, and save `CertificateSigningRequest.certSigningRequest`.
3. Go to <https://developer.apple.com/account/resources/certificates/add>, select **Developer ID Application**, upload the `.certSigningRequest`, download `developerID_application.cer`, and double-click it to add it to your `login` Keychain.

---

## Step 3: Create an App Store Connect API Key for Automated Notarization

Using an App Store Connect API Key (`.p8`) is cleaner and more reliable than app-specific passwords because it never triggers 2FA prompts in CI.

1. Visit <https://appstoreconnect.apple.com/access/integrations/api>.
2. Under **Team Keys**, click **`+` (Generate API Key)**:
   - **Name:** `nev2prores GitHub Notarization`
   - **Access / Role:** `Developer` (or `Admin`)
3. After creation, record three items:
   - **Issuer ID** (UUID at the top of the page, e.g. `12345678-90ab-cdef-1234-567890abcdef`)
   - **Key ID** (10-character ID next to your key, e.g. `9X8Y7Z6W5V`)
   - Click **Download API Key** to save `AuthKey_9X8Y7Z6W5V.p8` somewhere safe outside the Git repository (Apple only lets you download this file **once**).

---

## Step 4: Sign & Notarize Locally on Your Mac (Test First!)

Store your notarization credentials once in your local macOS Keychain:

```bash
xcrun notarytool store-credentials "nev2prores-notary" \
  --key "$HOME/.private_keys/AuthKey_9X8Y7Z6W5V.p8" \
  --key-id "9X8Y7Z6W5V" \
  --issuer "12345678-90ab-cdef-1234-567890abcdef"
```

Now build, sign (Hardened Runtime), notarize, and staple `dist/nev2prores.dmg` in a single command:

```bash
APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID1234)" \
APPLE_NOTARY_KEYCHAIN_PROFILE="nev2prores-notary" \
./scripts/bundle_macos_app.sh
```

Verify that macOS Gatekeeper fully accepts both the `.app` and the `.dmg`:

```bash
codesign -dv --verbose=4 dist/nev2prores.app
xcrun stapler validate dist/nev2prores.dmg
spctl -a -t open --context context:primary-signature -v dist/nev2prores.dmg
```

Expected output:
```text
dist/nev2prores.dmg: accepted
source=Notarized Developer ID
```

---

## Step 5: Configure GitHub Actions Secrets for Automated Releases

So that pushing a Git tag like `git tag v0.2.0 && git push origin v0.2.0` automatically builds, signs, notarizes, staples, and publishes `nev2prores.dmg` on GitHub Releases:

### 5.1 Export Your Certificate as a Password-Protected `.p12`
1. Open **Keychain Access → My Certificates**.
2. Expand **Developer ID Application: Your Name (TEAMID1234)** so both the certificate and its private key are selected, right-click → **Export 2 items…**.
3. Save as `Certificates.p12` and set a strong export password (save this password for `APPLE_CERTIFICATE_PASSWORD`).
4. Base64-encode both `Certificates.p12` and your `AuthKey_<KEYID>.p8` onto your clipboard (no line wraps):
   ```bash
   # Copy base64 of Certificates.p12:
   base64 -i Certificates.p12 | tr -d '\n' | pbcopy

   # Copy base64 of AuthKey_9X8Y7Z6W5V.p8:
   base64 -i "$HOME/.private_keys/AuthKey_9X8Y7Z6W5V.p8" | tr -d '\n' | pbcopy
   ```
5. Delete the temporary `Certificates.p12` file from disk once uploaded (`rm -f Certificates.p12`).

### 5.2 Add the 6 Repository Secrets on GitHub
Open **<https://github.com/adrienledoux-dev/nev2prores/settings/secrets/actions>** → **New repository secret**, and add:

| GitHub Secret Name | Example / Value |
|---|---|
| `APPLE_SIGNING_IDENTITY` | `Developer ID Application: Adrien Ledoux (A1B2C3D4E5)` |
| `APPLE_CERTIFICATE_P12_BASE64` | Output of `base64 -i Certificates.p12 \| tr -d '\n'` |
| `APPLE_CERTIFICATE_PASSWORD` | Password you chose when exporting `Certificates.p12` |
| `APPLE_API_KEY_ID` | `9X8Y7Z6W5V` |
| `APPLE_API_ISSUER_ID` | `12345678-90ab-cdef-1234-567890abcdef` |
| `APPLE_API_PRIVATE_KEY_BASE64` | Output of `base64 -i AuthKey_9X8Y7Z6W5V.p8 \| tr -d '\n'` |

> **Graceful Fallback:** If these secrets are not yet configured, `.github/workflows/release.yml` still succeeds and publishes an ad-hoc signed `nev2prores.dmg` alongside `SHA256SUMS.txt` (users without notarization can clear quarantine once via `xattr -dr com.apple.quarantine /Applications/nev2prores.app`). As soon as the secrets above are added, the next tagged release (`v0.2.1`, etc.) will automatically be Developer ID signed, notarized, and stapled.

---

## Note on External `NEV2PRORES_TICORAW_DYLIB` Plugins Under Hardened Runtime

When `nev2prores.app` is signed with the Hardened Runtime (`--options runtime`) **without** library-validation relaxation entitlements, macOS blocks `dlopen()` of unsigned third-party `.dylib` plugins. If you plan to load a custom `NEV2PRORES_TICORAW_DYLIB` library inside the notarized GUI bundle:
- Either sign the `.dylib` with the **same Team ID** (`Developer ID Application`), or
- Add an `entitlements.plist` containing `<key>com.apple.security.cs.disable-library-validation</key><true/>` and pass `--entitlements entitlements.plist` to `codesign`.
