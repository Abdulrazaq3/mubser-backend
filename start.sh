#!/bin/sh
# start.sh - Railway startup script

# استخدام PORT من البيئة أو 8000 كافتراضي
PORT="${PORT:-8000}"

echo "🚀 Starting Mubser Backend on port $PORT"

exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
