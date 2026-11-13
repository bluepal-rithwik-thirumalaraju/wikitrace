#!/usr/bin/env bash
# Install only the small command-line prerequisites in the disposable E2B VM.
set -euo pipefail

mkdir -p "$HOME/.local/bin"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
command -v uv >/dev/null 2>&1 || { echo "uv install failed" >&2; exit 1; }

node_major=0
if command -v node >/dev/null 2>&1; then
  node_major="$(node -p 'process.versions.node.split(".")[0]')"
fi
if [ "$node_major" -lt 20 ]; then
  case "$(uname -m)" in
    x86_64) node_arch=x64 ;;
    aarch64) node_arch=arm64 ;;
    *) echo "unsupported Node architecture: $(uname -m)" >&2; exit 1 ;;
  esac
  node_version=v22.16.0
  node_dir="$HOME/.local/node-$node_version-linux-$node_arch"
  if [ ! -x "$node_dir/bin/node" ]; then
    curl -fsSL "https://nodejs.org/dist/$node_version/node-$node_version-linux-$node_arch.tar.xz" \
      -o /tmp/wikitrace-node.tar.xz
    tar -xJf /tmp/wikitrace-node.tar.xz -C "$HOME/.local"
  fi
  ln -sfn "$node_dir/bin/node" "$HOME/.local/bin/node"
  ln -sfn "$node_dir/bin/npm" "$HOME/.local/bin/npm"
fi

uv --version
node --version
npm --version
