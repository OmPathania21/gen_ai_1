#!/usr/bin/env bash
# Start VERITAS at http://localhost:8000 (macOS / Linux). First run: python3 scripts/setup.py
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Setup has not been run yet. Run:  python3 scripts/setup.py"; exit 1
fi
if ! grep -qE '^GROQ_API_KEY=gsk_' .env 2>/dev/null || grep -q 'gsk_your_key_here' .env; then
  echo "Warning: no Groq API key in .env. Questions will fail until you add one (see README, step 3)."
fi
if command -v npm >/dev/null 2>&1; then
  if [ ! -d web/node_modules ]; then (cd web && npm install); fi
  (cd web && npm run build)
elif [ ! -f web/dist/index.html ]; then
  echo "Warning: Node.js is not installed, so the web page cannot be built (API only)."
fi
echo "VERITAS is starting at http://localhost:${PORT:-8000} (press Ctrl+C to stop)"
exec .venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
