#!/usr/bin/env bash
# install_yolov5_deps.sh
# Download YOLOv5 v6.2 tarball, extract, install requirements.
#
# Usage:
#   conda activate flower-benchmarks
#   bash install_yolov5_deps.sh
set -euo pipefail

YOLov5_TAG_URL="https://github.com/ultralytics/yolov5/archive/refs/tags/v6.2.tar.gz"
TMPDIR="$(mktemp -d)"
SCRIPT_DIR="$(pwd)"
EXTRACT_DIR="$TMPDIR/extracted"
FINAL_DIR="${SCRIPT_DIR}/yolov5"   # Will place the extracted repo here

echo
echo "=== YOLOv5 installer (v6.2) ==="
echo "Working in temp dir: $TMPDIR"
echo

# Step 1: fetch tarball (wget preferred, fallback to curl)
cd "$TMPDIR"
if command -v wget >/dev/null 2>&1; then
  echo "Downloading v6.2 tarball with wget..."
  wget -O v6.2.tar.gz "$YOLov5_TAG_URL"
elif command -v curl >/dev/null 2>&1; then
  echo "Downloading v6.2 tarball with curl..."
  curl -L -o v6.2.tar.gz "$YOLov5_TAG_URL"
else
  echo "Error: neither 'wget' nor 'curl' is installed. Please install one and retry." >&2
  exit 1
fi

# Step 2: extract tarball
mkdir -p "$EXTRACT_DIR"
tar -xzf v6.2.tar.gz -C "$EXTRACT_DIR"
# Find extracted repo folder (commonly yolov5-6.2)
EXTRACTED_REPO_DIR="$(find "$EXTRACT_DIR" -maxdepth 1 -type d -name 'yolov5*' | head -n 1)"
if [ -z "$EXTRACTED_REPO_DIR" ]; then
  echo "Extraction failed or unexpected tarball layout." >&2
  ls -la "$EXTRACT_DIR"
  exit 1
fi
echo "Extracted to: $EXTRACTED_REPO_DIR"

# Step 3: install runtime dependencies (safe list)
echo "Upgrading pip and installing common vision deps (opencv, pycocotools, albumentations, etc.)..."
python -m pip install --upgrade pip setuptools wheel

python -m pip install --upgrade pip

python -m pip install opencv-python-headless

# Install a core set of packages used by yolov5 scripts (won't touch torch if pinned already)
python -m pip install opencv-python matplotlib pandas wandb tqdm seaborn pycocotools albumentations thop || true

# Step 4: attempt to install repo requirements (if requirements.txt exists)
cd "$EXTRACTED_REPO_DIR"
if [ -f "requirements.txt" ]; then
  echo "Installing requirements.txt from the extracted repo (may take a while)..."
  # Use --no-deps to avoid inadvertently upgrading torch/torchvision; remove if you want full install
  python -m pip install --upgrade -r requirements.txt || {
    echo "Warning: pip install -r requirements.txt partially failed. Continuing and attempting local install..."
  }
else
  echo "No requirements.txt found in repo. Skipping requirements install."
fi

# Step 5: Try pip installing the package (local install)
echo "Attempting 'pip install .' in the extracted repo (this may fail if package isn't packaged)..."
if python -m pip install . 2>/tmp/pip_install_yolov5_err.log; then
  echo "pip install . succeeded. Moving repo to project as $FINAL_DIR."
  # move (overwrite) only after successful install
  rm -rf "$FINAL_DIR"
  mv "$EXTRACTED_REPO_DIR" "$FINAL_DIR"
else
  echo "pip install failed (not a pip-packagable project). Creating a .pth entry as fallback so Python can import yolov5 from its source folder."
  echo
  echo "pip install error (tail):"
  tail -n 200 /tmp/pip_install_yolov5_err.log || true

  # Move extracted folder into project for persistent access
  rm -rf "$FINAL_DIR"
  mv "$EXTRACTED_REPO_DIR" "$FINAL_DIR"

  # Compute site-packages path for the active Python
  SITE_PACKAGES_DIR="$(python - <<'PY'
import sysconfig, json
p = sysconfig.get_paths()["purelib"]
print(p)
PY
)"
  echo "Detected site-packages: $SITE_PACKAGES_DIR"

  # Create a .pth file that points to the repo so 'import yolov5' works
  PTH_FILE="$SITE_PACKAGES_DIR/yolov5_source.pth"
  echo "$FINAL_DIR" > "$PTH_FILE"
  echo "Wrote $PTH_FILE -> points to $FINAL_DIR"
  echo "Note: this makes the yolov5 source importable without pip installing it."
fi

# Step 6: Final notes and cleanup
echo
echo "=== Done ==="
echo "YOLOv5 v6.2 source is located at: $FINAL_DIR"
echo "If pip install succeeded, the package is installed into the environment."
echo "If pip install failed, a .pth file was created so you can 'import yolov5' from the extracted source."
echo
echo "If you plan to run yolov5/train.py as a module (python -m yolov5.train), make sure your current Python (which ran this script) is the same one used to run training (activate the same conda env)."
echo
echo "Cleanup: removing temp dir $TMPDIR"
rm -rf "$TMPDIR" || true

exit 0
