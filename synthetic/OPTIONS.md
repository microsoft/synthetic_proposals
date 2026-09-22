# Synthetic Grounded Proposal Generation

This directory contains a paper-grounded generator for synthetic NSF proposals. It reads the same proposal-generation test parquet used by `baselines/ground_truth_proposal.py`, searches arXiv using PI names plus award title/abstract terms, extracts text from likely resulting papers, and asks an Azure OpenAI model to produce a proposal in the same `generated_proposal` schema.

The main output is JSONL. Each row preserves the baseline-compatible fields:

- `sample_id`, `row_index`, `award_id`, `title`, `abstract`, `por`
- `source_row`, `ground_truth`
- `generated_proposal`, `raw_response`

It also adds `grounding_papers`, `candidate_papers`, and `generation_prompt` so you can audit which papers grounded each proposal.

## Run

From the `creative_rl` repo root:

```bash
bash synthetic/run_synthetic_proposals.sh \
  --input-file data/day_night/proposal_gen/test.parquet \
  --max-samples 10
```

By default the runner writes to:

```text
output_dir/creative_rl/proposal_gen/gpt51_synthetic_grounded_proposals.jsonl
```

The script is resumable. Existing `sample_id` rows in the output JSONL are skipped unless you pass `--overwrite`.

## Paper Discovery

For each award, the script:

1. Builds arXiv queries from PI names, award title terms, and abstract terms.
2. Scores candidates by PI-author overlap, topical overlap, and publication date proximity to the award period.
3. Downloads each promising arXiv PDF and extracts text with `arxiv2text` when available.
4. Falls back to `pdftotext`, then `pypdf` or `PyPDF2` if installed.
5. Looks for explicit NSF award evidence, including award IDs and acknowledgement/funding text.
6. Keeps the top 2-3 papers above `--min-paper-score`; samples with no papers are skipped by default.

Useful options:

```bash
--papers-per-award 3
--paper-search-limit 20
--min-paper-score 8.0
--allow-no-papers
--dry-run
```

Install `arxiv2text` in the same environment used to run the script for best extraction quality. The script can still run with the fallbacks if `arxiv2text` is unavailable.
