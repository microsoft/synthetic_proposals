#!/usr/bin/env bash
# Serve the arXiv dense index on :8000/retrieve.
# Leave this running: night_ai_scientist training calls it on every `search` action.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INDEX_DIR="${ARXIV_INDEX_DIR:-$HERE/../output/arxiv_index}"
CORPUS="${ARXIV_CORPUS:-$HERE/../output/arxiv_papers.jsonl}"
RETRIEVER_NAME="${RETRIEVER_NAME:-minilm}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

python "$HERE/arxiv_emb_retrieval_server.py" \
    --index_path "$INDEX_DIR/${RETRIEVER_NAME}_Flat.index" \
    --corpus_path "$CORPUS" \
    --topk 1 \
    --retriever_name "$RETRIEVER_NAME" \
    --retriever_model "$RETRIEVER_MODEL" \
    --faiss_gpu
