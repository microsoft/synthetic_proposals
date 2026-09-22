#!/usr/bin/env bash
# Generate grounded synthetic proposals. See ../README.md and ../.env.example.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${SYNTHETIC_PROPOSAL_AZURE_ENDPOINT:?not set — see .env.example}"
: "${CLIENT_ID:?not set — export the managed-identity client ID that can reach your Azure OpenAI resource (see .env.example)}"

MODEL="${SYNTHETIC_PROPOSAL_AZURE_MODEL:-gpt-4.1}"
OUT="${SYNTHETIC_PROPOSAL_OUTPUT_FILE:-$HERE/../output/synthetic_grounded_proposals.jsonl}"
mkdir -p "$(dirname "$OUT")"

python "$HERE/generate_synthetic_proposals.py" \
  --azure-endpoint "${SYNTHETIC_PROPOSAL_AZURE_ENDPOINT}" \
  --model "${MODEL}" \
  --output-file "${OUT}" \
  --max-tokens 5120 \
  --reasoning-effort minimal \
  --reasoning-budget-tokens 8192 \
  "$@"
