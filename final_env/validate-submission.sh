#!/usr/bin/env bash
set -uo pipefail

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "Usage: %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi

echo "Starting Validation..."

# 1. Ping the HF Space
echo -e "\n[1/3] Pinging Hugging Face Space: ${PING_URL}"
status_code=$(curl -o /dev/null -s -w "%{http_code}\n" "${PING_URL}")
if [ "$status_code" -eq 200 ] || [ "$status_code" -eq 405 ] || [ "$status_code" -eq 404 ]; then
    echo "✓ Space is reachable."
else
    echo "✗ Space returned HTTP $status_code. Ensure your Space is running."
    exit 1
fi

# 2. Check OpenEnv Spec
echo -e "\n[2/3] Validating OpenEnv Spec locally..."
if openenv validate "${REPO_DIR}"; then
    echo "✓ openenv validate passed."
else
    echo "✗ openenv validate failed."
    exit 1
fi

# 3. Docker Build Test
echo -e "\n[3/3] Testing Docker Build..."
if docker build -t openenv-eval-test -f ${REPO_DIR}/server/Dockerfile ${REPO_DIR}; then
    echo "✓ Docker image built successfully."
else
    echo "✗ Docker build failed."
    exit 1
fi

echo -e "\nAll pre-submission checks passed!"
