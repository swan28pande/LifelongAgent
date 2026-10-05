#!/usr/bin/env bash
# Start TiMem self-hosted databases (Postgres + Qdrant)
# After this, start the API server manually — see below.
set -e

cd "$(dirname "$0")"

echo "=== Starting database containers ==="
cd migration
docker compose up -d
cd ..

echo "=== Waiting for Postgres to be ready ==="
until docker exec timem_demo_postgres pg_isready -U timem_user -d timem_db 2>/dev/null; do
    sleep 2
done
echo "Postgres ready."

echo "=== Waiting for Qdrant to be ready ==="
until curl -sf http://localhost:16333/healthz >/dev/null 2>&1; do
    sleep 2
done
echo "Qdrant ready."

echo ""
echo "=== Databases are up. Now start the API server: ==="
echo "  cd baselines/TiMem"
echo "  PYTHONPATH=. uvicorn services.memory_generation_service:create_memory_generation_app --factory --port 8001"
