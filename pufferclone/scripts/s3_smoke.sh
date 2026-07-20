#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

minio_container=${MINIO_CONTAINER:-pufferclone-minio}
minio_network=${MINIO_NETWORK:-pufferclone-minio-net}
mc_image=${MINIO_MC_IMAGE:-minio/mc:latest}
s3_url=${PUFFERCLONE_S3_URL:-http://127.0.0.1:9000}
s3_bucket=${PUFFERCLONE_S3_BUCKET:-pufferclone-smoke}
access_key=${AWS_ACCESS_KEY_ID:-minioadmin}
secret_key=${AWS_SECRET_ACCESS_KEY:-minioadmin}
namespace="s3-smoke-$$"
api_port=8666
api_url="http://127.0.0.1:$api_port"
server_pid=
started_minio=0
log_file=$(mktemp "${TMPDIR:-/tmp}/pufferclone-s3-smoke.XXXXXX")

port_is_open() {
    (echo >/dev/tcp/127.0.0.1/"$api_port") >/dev/null 2>&1
}

dump_server_log() {
    echo "pufferclone S3 server failed to become ready; log:" >&2
    cat "$log_file" >&2
}

cleanup() {
    status=$?
    if [[ -n "$server_pid" ]]; then
        curl -fsS -X DELETE "$api_url/v1/namespaces/$namespace" >/dev/null 2>&1 || true
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
    fi
    rm -f "$log_file"
    if [[ "$started_minio" -eq 1 ]]; then
        docker compose down >/dev/null
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

if ! command -v lsof >/dev/null 2>&1; then
    echo "s3 smoke requires lsof to verify API socket ownership" >&2
    exit 1
fi
if port_is_open; then
    echo "refusing to start S3 smoke server: port $api_port is already occupied" >&2
    exit 1
fi

if [[ "$(docker inspect --format '{{.State.Running}}' "$minio_container" 2>/dev/null || true)" != "true" ]]; then
    docker compose up -d minio
    started_minio=1
fi

for _ in {1..30}; do
    if curl -fsS "$s3_url/minio/health/live" >/dev/null; then
        break
    fi
    sleep 1
done
curl -fsS "$s3_url/minio/health/live" >/dev/null

run_mc_shell() {
    local name="pufferclone-mc-$$-$RANDOM"
    docker run --rm --name "$name" --network "$minio_network" --entrypoint /bin/sh "$mc_image" -c "$1"
}

run_mc_shell "mc alias set local http://minio:9000 '$access_key' '$secret_key' >/dev/null && mc mb --ignore-existing 'local/$s3_bucket' >/dev/null"

cargo build --quiet --bin pufferclone
PUFFERCLONE_S3_URL="$s3_url" \
PUFFERCLONE_S3_BUCKET="$s3_bucket" \
AWS_ACCESS_KEY_ID="$access_key" \
AWS_SECRET_ACCESS_KEY="$secret_key" \
target/debug/pufferclone >"$log_file" 2>&1 &
server_pid=$!

ready=0
for _ in {1..30}; do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        dump_server_log
        exit 1
    fi
    listener_pids=$(lsof -nP -t -iTCP:"$api_port" -sTCP:LISTEN 2>/dev/null || true)
    if [[ -n "$listener_pids" ]] && ! grep -qx "$server_pid" <<<"$listener_pids"; then
        echo "S3 smoke API port $api_port is owned by an unexpected process: $listener_pids" >&2
        dump_server_log
        exit 1
    fi
    if grep -qx "$server_pid" <<<"$listener_pids" \
        && curl -fsS "$api_url/v1/namespaces" >/dev/null; then
        ready=1
        break
    fi
    sleep 1
done
if [[ "$ready" -ne 1 ]]; then
    dump_server_log
    exit 1
fi

upsert_response=$(curl -fsS -X POST \
    -H 'content-type: application/json' \
    "$api_url/v1/namespaces/$namespace" \
    -d '{"upserts":[{"id":"smoke-doc","vector":[1,0],"attributes":{"text":"minio smoke"}}],"schema":{"text":{"full_text_search":true}}}')
[[ "$upsert_response" == *'"upserted":1'* ]]

query_response=$(curl -fsS -X POST \
    -H 'content-type: application/json' \
    "$api_url/v1/namespaces/$namespace/query" \
    -d '{"vector":[1,0],"top_k":1}')
[[ "$query_response" == *'"id":"smoke-doc"'* ]]

echo "S3 smoke passed for namespace $namespace"
