#!/usr/bin/env bash
# setup_torchcodec_mac.sh
# macOS setup for TorchCodec + FFmpeg 7 with a uv/venv project.
# - Builds FFmpeg 7.1.2 to ~/.local/ffmpeg7 (user-local, no sudo)
# - Patches TorchCodec libs with LC_RPATH to find FFmpeg + PyTorch
# - Codesigns edited binaries
# - Smoke tests import + a WAV save via torchaudio

set -euo pipefail

# ---------- Config for user-local installation ----------
: "${FFMPEG_PREFIX:=$HOME/.local/ffmpeg7}"      # User-local installation
: "${FFMPEG_TAG:=n7.1.2}"
: "${FFMPEG_SRC:=$FFMPEG_PREFIX/src}"           # Source inside install dir

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  : "${VENV:=$VIRTUAL_ENV}"
  echo "==> Using activated venv: $VENV"
else
  : "${VENV:=$(pwd)/.venv}"
  echo "==> Using current directory's venv: $VENV"
fi

# Ensure directories exist
mkdir -p "$FFMPEG_PREFIX"

TC_DIR="$VENV/lib/python3.10/site-packages/torchcodec"
TORCH_LIB="$VENV/lib/python3.10/site-packages/torch/lib"
FFMPEG_LIB="$FFMPEG_PREFIX/lib"
FFMPEG_BIN="$FFMPEG_PREFIX/bin/ffmpeg"

green(){ printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
red(){ printf "\033[31m%s\033[0m\n" "$*"; }

need() { command -v "$1" >/dev/null 2>&1 || { red "Missing: $1"; exit 1; }; }

# ---------- Pre-flight ----------
green "==> Pre-flight checks"
need xcode-select
need git
need make
need install_name_tool
need codesign
need otool

# ---------- Build FFmpeg 7 (if missing) ----------
if [[ -x "$FFMPEG_BIN" && -f "$FFMPEG_LIB/libavformat.61.dylib" ]]; then
  green "==> FFmpeg already present at $FFMPEG_PREFIX"
  "$FFMPEG_BIN" -version | head -n 1 || true
else
  yellow "==> Cloning FFmpeg into: $FFMPEG_SRC"
  if [[ ! -d "$FFMPEG_SRC/.git" ]]; then
    git clone https://git.ffmpeg.org/ffmpeg.git "$FFMPEG_SRC"
  fi
  pushd "$FFMPEG_SRC" >/dev/null
  git fetch --tags
  git checkout "$FFMPEG_TAG"

  yellow "==> Configuring FFmpeg (shared libs) -> $FFMPEG_PREFIX"
  ./configure \
    --prefix="$FFMPEG_PREFIX" \
    --enable-shared         \
    --disable-static        \
    --enable-pthreads

  yellow "==> Building FFmpeg..."
  make -j"$(sysctl -n hw.ncpu)"

  yellow "==> Installing FFmpeg (user-local, no sudo)..."
  make install
  popd >/dev/null

  green "==> FFmpeg installed to user directory:"
  "$FFMPEG_BIN" -version | head -n 1

  # Add to PATH
  if ! echo "$PATH" | grep -q "$FFMPEG_PREFIX/bin"; then
    yellow "   To use ffmpeg from anywhere, add to your shell config:"
    yellow "   export PATH=\"$FFMPEG_PREFIX/bin:\$PATH\""
  fi
fi

# ---------- Sanity: venv + torchcodec files exist ----------
green "==> Checking venv + torchcodec paths"
[[ -d "$VENV" ]] || { red "VENV not found: $VENV (activate your uv venv or set VENV=/path/to/.venv)"; exit 1; }
[[ -d "$TC_DIR" ]] || { red "TorchCodec not found in venv: $TC_DIR (uv add torchcodec==0.8.*)"; exit 1; }
[[ -d "$TORCH_LIB" ]] || { red "Torch lib dir missing: $TORCH_LIB"; exit 1; }
[[ -d "$FFMPEG_LIB" ]] || { red "FFMPEG lib dir missing: $FFMPEG_LIB"; exit 1; }

# ---------- Patch TorchCodec rpaths ----------
green "==> Patching TorchCodec LC_RPATH (FFmpeg + Torch)"
shopt -s nullglob
PATCH_TARGETS=(
  "$TC_DIR"/libtorchcodec_core*.dylib
  "$TC_DIR"/libtorchcodec_custom_ops*.dylib
  "$TC_DIR"/libtorchcodec_pybind_ops7.so
)

for f in "${PATCH_TARGETS[@]}"; do
  yellow "  add rpath -> $(basename "$f")"
  install_name_tool -add_rpath "$FFMPEG_LIB" "$f" 2>/dev/null || true
  install_name_tool -add_rpath "$TORCH_LIB"   "$f" 2>/dev/null || true
done

# ---------- Fix private libc++ if referenced ----------
green "==> Fixing private libc++ references (if any)"
for f in "${PATCH_TARGETS[@]}"; do
  if otool -L "$f" | grep -q '@loader_path/.dylibs/libc++.1.0.dylib'; then
    yellow "  libc++ -> $(basename "$f")"
    install_name_tool -change "@loader_path/.dylibs/libc++.1.0.dylib" "/usr/lib/libc++.1.dylib" "$f" || true
  fi
done

# ---------- Verify rpaths on core + pybind ----------
green "==> Verifying LC_RPATH is present"
echo "---- core7 ----"
otool -l "$TC_DIR/libtorchcodec_core7.dylib" | grep -A2 LC_RPATH || true
echo "---- pybind ----"
otool -l "$TC_DIR/libtorchcodec_pybind_ops7.so" | grep -A2 LC_RPATH || true

# ---------- Codesign edited binaries ----------
green "==> Codesigning modified TorchCodec binaries"
for f in "${PATCH_TARGETS[@]}"; do
  if [[ -f "$f" ]]; then
    codesign --force --sign - --timestamp=none "$f" 2>/dev/null && \
      echo "  ✓ Signed: $(basename "$f")" || \
      echo "  ⚠ Failed to sign: $(basename "$f")"
  fi
done

# ---------- WAV test via torchaudio ----------
green "==> WAV save test via torchaudio (TorchCodec backend)"
python - <<'PY'
import torch, torchaudio
x = torch.zeros(2, 16000, dtype=torch.float32)
torchaudio.save("probe.wav", x, 16000)
print("✅ torchaudio.save worked -> probe.wav")
PY

# ---------- Cleanup ----------
green "==> Cleaning up"
rm -f probe.wav

# Remove source directory if build succeeded (saves ~500MB)
if [[ -x "$FFMPEG_BIN" && -d "$FFMPEG_SRC" ]]; then
  SIZE=$(du -sh "$FFMPEG_SRC" 2>/dev/null | cut -f1)
  yellow "  FFmpeg source using $SIZE - removing..."
  rm -rf "$FFMPEG_SRC"
  echo "  ✓ Removed build files, kept installation in $FFMPEG_PREFIX"
fi

green "==> All done!"
green "  FFmpeg installed at: $FFMPEG_PREFIX"
green "  To uninstall later: rm -rf $FFMPEG_PREFIX"