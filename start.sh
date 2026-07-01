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

# Helper: ensure a model is available (downloads if needed)
ensure_model() {
  local label="$1"
  local raw_path="$2"
  local fallback_repo="$3"
  local path="${raw_path/#\~/$HOME}"  # expand tilde

  if [[ "$path" != /* ]] && [ -n "$path" ]; then
    # It's a HuggingFace repo ID — download into cache if not already present
    echo "📦  Checking $label ($path)..."
    huggingface-cli download "$path" --quiet || echo "    ⚠  Download failed for $label"
  elif [[ "$path" == /* ]] && [ ! -d "$path" ]; then
    echo "📦  $label not found at '$path', downloading $fallback_repo..."
    huggingface-cli download "$fallback_repo" --local-dir "$path" --quiet \
      || echo "    ⚠  Download failed for $label"
  fi
}

# Main model
MAIN_MODEL=$(grep '^MLX_MODEL_PATH=' .env | cut -d= -f2 | tr -d ' "')
ensure_model "main model" "$MAIN_MODEL" "mlx-community/Qwen3.5-9B-MLX-4bit"

# Small model (background tasks)
SMALL_MODEL=$(grep '^MLX_SMALL_MODEL_PATH=' .env | cut -d= -f2 | tr -d ' "')
if [ -n "$SMALL_MODEL" ]; then
  ensure_model "small model" "$SMALL_MODEL" "mlx-community/Qwen3-1.7B-4bit"
fi

# start Neo4j if not already running
NEO4J_CONTAINER="memgraph-neo4j"
NEO4J_VOLUME="memgraph-neo4j-data"
NEO4J_PASSWORD=$(grep NEO4J_PASSWORD .env | cut -d= -f2 | tr -d ' "' || echo "password")

if ! docker ps --format '{{.Names}}' | grep -q "^${NEO4J_CONTAINER}$"; then
  echo "🗄   Starting Neo4j..."
  docker run -d \
    --name "$NEO4J_CONTAINER" \
    -p 7474:7474 -p 7687:7687 \
    -v "${NEO4J_VOLUME}:/data" \
    -e NEO4J_AUTH="neo4j/${NEO4J_PASSWORD}" \
    neo4j:latest > /dev/null
  echo "    Waiting for Neo4j to be ready..."
  until docker exec "$NEO4J_CONTAINER" cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "RETURN 1" &>/dev/null; do
    sleep 2
  done
  echo "    Neo4j is ready."
else
  echo "🗄   Neo4j already running."
fi

# check python deps
python3 -c "import neo4j, mlx_lm, fastapi, uvicorn, qdrant_client" 2>/dev/null || {
  echo "📦  Installing dependencies..."
  pip install -r requirements.txt -q
}
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
echo "🚀  Starting server at http://localhost:8000"
echo "   Press Ctrl+C to stop."
echo ""
python3 server.py
