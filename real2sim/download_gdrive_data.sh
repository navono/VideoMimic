#!/bin/bash

set -e

# === CONFIGURATION ===
HF_REPO="Hongsuk/VideoMimic-Real2Sim-assets"   # HF dataset id
HF_FILE="assets.zip"       # file name in the repo (adjust if different)
TARGET_DIR="./"                                # where to put the files
ZIP_NAME="$HF_FILE"                            # local name of the downloaded zip

HF_URL="https://huggingface.co/datasets/${HF_REPO}/resolve/main/${HF_FILE}"

# === CHECK DEPENDENCIES ===
if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null
then
    echo "Error: neither wget nor curl is installed. Please install one of them."
    exit 1
fi

# === MAKE TARGET DIR ===
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# === DOWNLOAD ===
echo "⬇ Downloading file from Hugging Face..."
if command -v wget &> /dev/null; then
    wget -O "$ZIP_NAME" "$HF_URL"
else
    curl -L "$HF_URL" -o "$ZIP_NAME"
fi

# === UNZIP ===
echo "📦 Unzipping..."
unzip "$ZIP_NAME"

# === CLEAN UP ===
echo "🧹 Removing zip file..."
rm "$ZIP_NAME"

echo "✅ Done. Files are in $TARGET_DIR"

