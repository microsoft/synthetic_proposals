# Process arxiv-metadata-oai-snapshot.json to extract id, title, and abstract for
# each paper and save to arxiv_papers.jsonl in this format:
# {"id": "<arxiv_id>", "contents": "<title>\n<abstract>"}
#
# The input file is very large (5GB), so this script processes one line at a time
# and never loads the full file into memory.

import argparse
import json
from pathlib import Path
from tqdm import tqdm


DEFAULT_INPUT = "arxiv-metadata-oai-snapshot.json"
DEFAULT_OUTPUT = "arxiv_papers.jsonl"


def to_text(value: object) -> str:
	"""Convert a possibly missing/non-string JSON field into a normalized string."""
	if value is None:
		return ""
	if isinstance(value, str):
		return value.strip().replace("\n", " ")
	return str(value).strip().replace("\n", " ")


def convert_snapshot(input_path: Path, output_path: Path) -> tuple[int, int]:
	"""
	Convert the arXiv metadata snapshot to JSONL.

	Returns a tuple of (written_count, skipped_count).
	"""
	written = 0
	skipped = 0

	with input_path.open("r", encoding="utf-8") as infile, output_path.open(
		"w", encoding="utf-8"
	) as outfile:
		for line_num, line in tqdm(enumerate(infile, start=1)):
			line = line.strip()
			if not line:
				continue

			try:
				paper = json.loads(line)
			except json.JSONDecodeError:
				skipped += 1
				continue

			arxiv_id = to_text(paper.get("id"))
			title = to_text(paper.get("title"))
			abstract = to_text(paper.get("abstract"))

			if not arxiv_id:
				skipped += 1
				continue

			record = {
				"id": arxiv_id,
				"contents": f"{title}\n{abstract}",
			}
			outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
			written += 1

			if line_num % 1_000_000 == 0:
				print(f"Processed {line_num:,} lines; wrote {written:,} records...")

	return written, skipped


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Stream-convert arXiv metadata snapshot to paper contents JSONL."
	)
	parser.add_argument(
		"--input",
		default=DEFAULT_INPUT,
		help=f"Path to input snapshot file (default: {DEFAULT_INPUT})",
	)
	parser.add_argument(
		"--output",
		default=DEFAULT_OUTPUT,
		help=f"Path to output JSONL file (default: {DEFAULT_OUTPUT})",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_path = Path(args.input)
	output_path = Path(args.output)

	if not input_path.exists():
		raise FileNotFoundError(f"Input file not found: {input_path}")

	written, skipped = convert_snapshot(input_path=input_path, output_path=output_path)
	print(f"Done. Wrote {written:,} records to {output_path}. Skipped {skipped:,} lines.")


if __name__ == "__main__":
	main()
