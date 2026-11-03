#!/usr/bin/env bash
# Downloads the SciFact benchmark and the demo paper collection. Same as: python scripts/download_data.py
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 scripts/download_data.py
