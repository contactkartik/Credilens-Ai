#!/bin/sh
# All-in-one entrypoint: start uvicorn (background), then nginx in foreground.
set -e

# Ensure the SQLite directory exists (HF Spaces mounts /data; harmless elsewhere)
mkdir -p /data 2>/dev/null || true

# Start the FastAPI backend
cd /app/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Give uvicorn a moment before nginx starts proxying
sleep 2

# nginx in foreground keeps the container alive
exec nginx -g 'daemon off;'
