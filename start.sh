#!/usr/bin/env bash
# MemGraph Agent — local startup
set -e

echo "🧠  MemGraph Agent"
echo "──────────────────"

# check .env
if [ ! -f .env ]; then
  cp .env.example .env
  echo "⚠  Created .env from .env.example — edit it if needed."
fi

# check MLX model path
MODEL_PATH=$(grep MLX_MODEL_PATH .env | cut -d= -f2 | tr -d ' "' || echo "~/mlx-models/qwen3.5-9b")
MODEL_PATH="${MODEL_PATH/#\~/$HOME}"  # expand tilde

if [ ! -d "$MODEL_PATH" ]; then
  echo "📦  MLX model not found at '$MODEL_PATH'."
  echo "    Downloading mlx-community/Qwen3.5-9B-MLX-4bit..."
  huggingface-cli download mlx-community/Qwen3.5-9B-MLX-4bit \
    --local-dir "$MODEL_PATH"
fi

# check python deps
python3 -c "import neo4j, mlx_lm, fastapi, uvicorn" 2>/dev/null || {
  echo "📦  Installing dependencies..."
  pip install -r requirements.txt -q
}

echo "🚀  Starting server at http://localhost:8000"
echo "   Press Ctrl+C to stop."
echo ""
python3 server.py