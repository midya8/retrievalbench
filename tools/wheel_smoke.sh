#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
python -m venv "$OUT/build"
"$OUT/build/bin/python" -m pip -q install build
"$OUT/build/bin/python" -m build --wheel --outdir "$OUT/wheel" "$ROOT"
python -m venv "$OUT/env"
"$OUT/env/bin/python" -m pip -q install "$OUT"/wheel/*.whl
cd "$OUT"
"$OUT/env/bin/python" "$ROOT/tools/verify_installed.py" "$ROOT/docs/audit.json" "$ROOT"
