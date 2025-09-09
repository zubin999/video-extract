#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${WHISPER_MODEL:-small}"
MODEL_DIR="/app/models"
MODEL_PATH="$MODEL_DIR/ggml-${MODEL_NAME}.bin"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading Whisper model: $MODEL_NAME"
  MIRROR_DOMAIN="${HF_MIRROR:-https://hf-mirror.com}"
  MIRROR_URL="$MIRROR_DOMAIN/ggerganov/whisper.cpp/resolve/main/ggml-${MODEL_NAME}.bin"

  # More robust curl options: force HTTP/1.1, prefer IPv4, retry on errors
  CURL_OPTS=(
    -fL            # fail on HTTP errors, follow redirects
    --http1.1      # avoid some HTTP/2/ALPN issues in restricted networks
    -4             # prefer IPv4 in case IPv6 is broken
    --retry 5      # retry a few times
    --retry-delay 3
    --retry-all-errors
    --connect-timeout 10
  )

  TMP_FILE="$MODEL_PATH.part"
  echo "Downloading from mirror: $MIRROR_URL"
  if ! curl "${CURL_OPTS[@]}" "$MIRROR_URL" -o "$TMP_FILE"; then
    echo "Mirror download failed." >&2
    exit 1
  fi

  if [ ! -s "$TMP_FILE" ]; then
    echo "Downloaded file is empty or missing." >&2
    exit 1
  fi
  mv "$TMP_FILE" "$MODEL_PATH"
else
  echo "Model already exists: $MODEL_PATH"
fi

# Export path for the app
export WHISPER_MODEL_PATH="$MODEL_PATH"
