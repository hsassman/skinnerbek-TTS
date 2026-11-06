#!/bin/bash
# Skinnerbek Studio - double-click to start (macOS). On Linux run: ./start.command
# First run installs a private copy of Python and the voice models (about 9 GB in total).
set -u
cd "$(dirname "$0")" || exit 1

hold() { echo; read -r -p "Press Enter to close this window." _; exit 1; }

if [ ! -f scripts/start.py ]; then
  echo "This folder is incomplete. Unzip the download first, then run start.command from the unzipped folder."
  hold
fi

UV_VERSION=0.13.5
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)
    UV_TARGET=aarch64-apple-darwin
    UV_SHA256=743f85035a1996ecb350659e01cd9faea9709829be09b1b4b03c73eaf3c9cd73
    APP_HOME="$HOME/Library/Application Support/SkinnerbekStudio" ;;
  Darwin-x86_64)
    echo "Skinnerbek Studio needs a Mac with Apple silicon (M1 or newer)."
    echo "PyTorch, the engine it runs on, no longer supports Intel Macs."
    hold ;;
  Linux-x86_64)
    UV_TARGET=x86_64-unknown-linux-gnu
    UV_SHA256=f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f
    APP_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/SkinnerbekStudio"
    export UV_TORCH_BACKEND=cpu ;;
  Linux-aarch64)
    UV_TARGET=aarch64-unknown-linux-gnu
    UV_SHA256=4bb69cf95ed13f23713bb7bc09f241896ea0c75b7c557e615a20604b82f9c403
    APP_HOME="${XDG_DATA_HOME:-$HOME/.local/share}/SkinnerbekStudio"
    export UV_TORCH_BACKEND=cpu ;;
  *)
    echo "This computer ($(uname -s) $(uname -m)) is not supported."
    hold ;;
esac

UV_DIR="$APP_HOME/uv-$UV_VERSION"
UV="$UV_DIR/uv"
VENV="$APP_HOME/env"
PY="$VENV/bin/python"
export UV_PYTHON_INSTALL_DIR="$APP_HOME/python"
export UV_CACHE_DIR="$APP_HOME/cache"
export UV_NO_CONFIG=1
export UV_PYTHON_PREFERENCE=only-managed

echo
echo "  Skinnerbek Studio"
echo "  -----------------"
echo

sha256() { if command -v shasum >/dev/null; then shasum -a 256 "$1"; else sha256sum "$1"; fi | cut -d' ' -f1; }

if [ -x .venv/bin/python ]; then
  # A developer checkout with its own .venv runs as-is.
  PY="$PWD/.venv/bin/python"
else
  if [ ! -x "$UV" ]; then
    echo "Setting up for the first time. This needs an internet connection."
    mkdir -p "$UV_DIR" || { echo "Could not create $UV_DIR."; hold; }
    TARBALL="$UV_DIR/uv.tar.gz"
    echo "Downloading the installer tool (uv $UV_VERSION)..."
    if ! curl -fsSL --retry 3 -o "$TARBALL" \
        "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-$UV_TARGET.tar.gz"; then
      rm -f "$TARBALL"
      echo "Download failed. Check your internet connection and run start again."
      hold
    fi
    if [ "$(sha256 "$TARBALL")" != "$UV_SHA256" ]; then
      rm -f "$TARBALL"
      echo "The downloaded installer did not match its published checksum, so it was deleted."
      echo "Run start again. If this keeps happening, your network may be altering downloads."
      hold
    fi
    tar -xzf "$TARBALL" -C "$UV_DIR" --strip-components=1 && rm -f "$TARBALL"
    [ -x "$UV" ] || { echo "Could not unpack the installer tool."; hold; }
  fi

  if [ ! -x "$PY" ]; then
    echo "Installing Python 3.12 for Skinnerbek Studio (does not touch any other Python)..."
    "$UV" venv --quiet --python 3.12 "$VENV" || { echo "Python could not be installed. Check your internet connection and run start again."; hold; }
  fi

  if ! cmp -s requirements.txt "$VENV/installed-requirements.txt"; then
    echo "Installing the app's components (about 1.5 GB, first run only)..."
    if ! "$UV" pip install --python "$PY" -r requirements.txt; then
      echo "Installing components failed. Check your internet connection and free disk space,"
      echo "then run start again. It continues where it stopped."
      hold
    fi
    cp requirements.txt "$VENV/installed-requirements.txt"
  fi
fi

"$PY" scripts/start.py || { echo; echo "Skinnerbek Studio stopped with an error. The message above says why."; hold; }
