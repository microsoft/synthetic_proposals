#!/usr/bin/env bash
# Embed the arXiv corpus into a FAISS index. Wants a multi-GPU node.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CORPUS="${ARXIV_CORPUS:-$HERE/../output/arxiv_papers.jsonl}"
SAVE_DIR="${ARXIV_INDEX_DIR:-$HERE/../output/arxiv_index}"
RETRIEVER_NAME="${RETRIEVER_NAME:-minilm}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

# faiss_type HNSW32/64/128 for ANN indexing; retriever_name bm25 for BM25.
python "$HERE/index_builder.py" \
    --retrieval_method "$RETRIEVER_NAME" \
    --model_path "$RETRIEVER_MODEL" \
    --corpus_path "$CORPUS" \
    --save_dir "$SAVE_DIR" \
    --use_fp16 \
    --max_length 512 \
    --batch_size 1024 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
