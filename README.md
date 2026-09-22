# Synthetic Proposals & arXiv Retrieval

Companion services for [night_ai_scientist](https://github.com/pkargupta/night_ai_scientist),
the Night Science agentic-creativity framework. Both are standalone — neither imports from
that repository — and they are split out here so the training repo stays focused on the
method itself.

| | |
|---|---|
| [`retrieval/`](retrieval/) | Dense arXiv retrieval service. Backs the `search` action during training. |
| [`verl_tool/`](verl_tool/) | The verl-side client for that service. Copied into the verl tree by night_ai_scientist's `setup.sh`. |
| [`synthetic/`](synthetic/) | Grounded synthetic proposal generation. Produces training data and reference proposals. |

```bash
pip install -r requirements.txt
cp .env.example .env        # then edit
set -a; . ./.env; set +a
```

## retrieval/ — arXiv retrieval service

The `search` action queries a local dense index. Download the arXiv metadata snapshot
([Kaggle: Cornell-University/arxiv](https://www.kaggle.com/datasets/Cornell-University/arxiv)),
then build the index and serve it:

```bash
python retrieval/process_arxiv.py \
    --input arxiv-metadata-oai-snapshot.json \
    --output output/arxiv_papers.jsonl

bash retrieval/build_index.sh      # embeds ~2.5M abstracts; wants a multi-GPU node
bash retrieval/arxiv_emb_retrieval.sh
```

`process_arxiv.py` streams the 5GB snapshot one line at a time, emitting
`{"id": "<arxiv_id>", "contents": "<title>\n<abstract>"}`. `build_index.sh` embeds it with
`all-MiniLM-L6-v2` into a FAISS `Flat` index. Set `ARXIV_CORPUS` and `ARXIV_INDEX_DIR` to
put them elsewhere; `RETRIEVER_NAME=bm25` uses the Pyserini/BM25 path instead.

**Leave the server running** — training calls it on every `search` action.

> A dead retrieval server does not crash training. `search` actions simply return nothing,
> and the model quietly learns that searching is worthless. Confirm the endpoint answers
> before launching a run.

### The interface

The service listens on **port 8000** and is the only contract between these two repos.
Anything honoring it will work:

```
POST /retrieve
  {"queries": ["..."], "topk": 3, "return_scores": false}
```

Point the training repo at it with `CREATIVE_RETRIEVER_URL`, or in
`examples/sglang_multiturn/config/tool_config/arxiv_search_tool_config.yaml`:

```yaml
retrieval_service_url: http://127.0.0.1:8000/retrieve
```

## verl_tool/ — the verl-side search client

```
verl_tool/verl/tools/arxiv_search_tool.py        SearchTool, registered via tool_config
verl_tool/verl/tools/utils/arxiv_utils.py        HTTP client with retries
```

Paths mirror their destination inside the verl tree, so night_ai_scientist's `setup.sh`
copies `verl_tool/.` over the checkout verbatim. **These two files are required** — the
training repo imports `verl.tools.utils.arxiv_utils` from its reward manager, so a build
without them will not start.

Unlike the rest of this repository they are not standalone: they import `verl.tools.base_tool`
and `verl.utils.rollout_trace`, and only make sense inside a verl checkout. They live here so
everything touching arXiv retrieval stays in one place.

## synthetic/ — grounded synthetic proposals

For each NSF award, find the arXiv papers that actually resulted from it, then have a model
write the proposal that would have preceded them. This yields *grounded* synthetic proposals
— useful as training data or as a reference point.

```bash
bash synthetic/run_synthetic_proposals.sh \
  --input-file /path/to/test.parquet \
  --max-samples 10
```

Papers are matched by PI-author overlap, topical overlap, publication date proximity to the
award period, and explicit NSF award-number acknowledgements in the PDF text.
[`synthetic/OPTIONS.md`](synthetic/OPTIONS.md) documents every flag and the output schema.

Requires `SYNTHETIC_PROPOSAL_AZURE_ENDPOINT` and `CLIENT_ID`; the script exits with a pointer
to `.env.example` if either is missing.

## Provenance

`retrieval/arxiv_emb_retrieval_server.py` and `retrieval/index_builder.py` are adapted from
the dense-retrieval tooling in [verl](https://github.com/volcengine/verl)'s `search_r1_like`
example and from [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)/LongRAG, modified to serve
an arXiv corpus.
