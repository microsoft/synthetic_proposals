import argparse
import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Any

import json_repair
import openai
import pandas as pd
from annotated_types import Len
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI
from pydantic import BaseModel, Field, ValidationError
from tqdm import tqdm
from typing_extensions import Annotated


DEFAULT_INPUT = Path("data/day_night/proposal_gen/test.parquet")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
DEFAULT_API_VERSION = "2024-12-01-preview"
DEFAULT_MODEL = "gpt-5.1"
DEFAULT_MAX_TOKENS = 5120
DEFAULT_RETRIES = 3
DEFAULT_PAPERS_PER_AWARD = 3
DEFAULT_SEARCH_LIMIT = 20
DEFAULT_TEXT_LIMIT = 24000
DEFAULT_ARXIV_MAX_RETRIES = 4
DEFAULT_ARXIV_BACKOFF_BASE = 2.0
DEFAULT_REASONING_EFFORT = "minimal"
DEFAULT_REASONING_BUDGET_TOKENS = 4096
ARXIV_API_URL = "https://export.arxiv.org/api/query"
APPROX_CHARS_PER_TOKEN = 4


class plan_schema(BaseModel):
    phase: str = Field(..., description="name or identifier of the research phase")
    idea: str = Field(..., description="detailed description of the idea for this research phase")
    experimental_plan: str = Field(..., description="detailed description of the experimental plan for this research phase")


class proposal_schema(BaseModel):
    proposal_title: str = Field(..., description="title of the proposal")
    proposal_summary: str = Field(
        ...,
        description="concise statement of what the proposal aims to do, why it is important, and what specific problems it will resolve",
    )
    background_and_significance: str = Field(
        ...,
        description="historical review of the field, including what has been done, what remains to be done, and how the proposal builds on existing knowledge",
    )
    research_plan: Annotated[list[plan_schema], Len(min_length=1, max_length=5)] = Field(
        ..., description="list of research phases within the proposal"
    )


class ProposalGenerationError(RuntimeError):
    def __init__(
        self,
        message: str,
        sample_id: str,
        last_raw_response: str = "",
        response_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.sample_id = sample_id
        self.last_raw_response = last_raw_response
        self.response_metadata = response_metadata or {}


@dataclass
class ArxivPaper:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str
    updated: str
    pdf_url: str
    entry_url: str
    search_query: str
    metadata_score: float = 0.0
    full_text_score: float = 0.0
    evidence_excerpt: str | None = None
    processed_text: str | None = None

    @property
    def total_score(self) -> float:
        return self.metadata_score + self.full_text_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic, paper-grounded NSF proposals from award metadata. "
            "Outputs JSONL rows compatible with baselines/ground_truth_proposal.py."
        )
    )
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--azure-endpoint", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--api-version", type=str, default=DEFAULT_API_VERSION)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=["minimal", "low", "medium", "high"],
        default=DEFAULT_REASONING_EFFORT,
        help="Reasoning effort for GPT-5 Azure chat completions.",
    )
    parser.add_argument(
        "--reasoning-budget-tokens",
        type=int,
        default=DEFAULT_REASONING_BUDGET_TOKENS,
        help=(
            "Extra completion-token budget reserved for hidden reasoning on GPT-5 models. "
            "Total max_completion_tokens = max_tokens + reasoning_budget_tokens."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--papers-per-award", type=int, default=DEFAULT_PAPERS_PER_AWARD)
    parser.add_argument("--paper-search-limit", type=int, default=DEFAULT_SEARCH_LIMIT)
    parser.add_argument("--min-paper-score", type=float, default=4.0)
    parser.add_argument("--arxiv-delay", type=float, default=3.0, help="Seconds to wait between arXiv API calls.")
    parser.add_argument("--arxiv-max-retries", type=int, default=DEFAULT_ARXIV_MAX_RETRIES)
    parser.add_argument("--arxiv-backoff-base", type=float, default=DEFAULT_ARXIV_BACKOFF_BASE)
    parser.add_argument("--paper-cache-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "paper_cache")
    parser.add_argument("--text-char-limit", type=int, default=DEFAULT_TEXT_LIMIT)
    parser.add_argument("--arxiv2text-command", type=str, default="arxiv2text")
    parser.add_argument(
        "--allow-no-papers",
        action="store_true",
        help="Generate from award metadata even when no likely arXiv papers are found. By default such samples are skipped.",
    )
    parser.add_argument(
        "--keep-failed-text",
        action="store_true",
        help="Keep failed or empty paper text cache files for debugging.",
    )
    return parser.parse_args()


def first_csv_env(env_name: str) -> str | None:
    raw_value = os.getenv(env_name, "").strip()
    if not raw_value:
        return None
    return raw_value.split(",")[0].strip() or None


def resolve_endpoint(cli_value: str | None) -> str | None:
    return cli_value or os.getenv("SYNTHETIC_PROPOSAL_AZURE_ENDPOINT") or first_csv_env("CREATIVE_AZURE_ENDPOINTS")


def resolve_model(cli_value: str | None) -> str:
    return cli_value or os.getenv("SYNTHETIC_PROPOSAL_AZURE_MODEL") or first_csv_env("CREATIVE_AZURE_MODELS") or DEFAULT_MODEL


def default_output_path(model: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
    return DEFAULT_OUTPUT_DIR / safe_model / "synthetic_grounded_proposals.jsonl"


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = unescape(value)
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("\r", " ").replace("\n", " ")
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def normalize_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            try:
                return json_repair.loads(value)
            except Exception:
                return {}
    return {}


def make_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "item"):
        try:
            return make_jsonable(value.item())
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): make_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return make_jsonable(value.tolist())
        except Exception:
            pass
    if pd.isna(value):
        return None
    return str(value)


def get_field(row: pd.Series, *field_names: str) -> Any:
    for field_name in field_names:
        if field_name not in row:
            continue
        value = row[field_name]
        try:
            if pd.isna(value):
                continue
        except ValueError:
            pass
        return value
    return None


def extract_por_text(por_value: Any) -> str | None:
    if por_value is None:
        return None
    if isinstance(por_value, str):
        return clean_text(por_value)
    if isinstance(por_value, dict):
        for key in ("por_txt_cntn", "por_cntn", "text", "content"):
            if por_value.get(key):
                return clean_text(por_value[key])
    return clean_text(por_value)


def extract_pi_names(pi_value: Any) -> list[str]:
    pi_value = make_jsonable(pi_value)
    if pi_value is None:
        return []
    pi_rows = pi_value if isinstance(pi_value, list) else [pi_value]
    names: list[str] = []
    for item in pi_rows:
        if not isinstance(item, dict):
            text = clean_text(item)
            if text:
                names.append(text)
            continue
        name = clean_text(item.get("pi_full_name"))
        if not name:
            first = clean_text(item.get("pi_first_name")) or ""
            last = clean_text(item.get("pi_last_name")) or ""
            name = clean_text(f"{first} {last}")
        if name:
            names.append(name)
    return list(dict.fromkeys(names))


def build_sample(row: pd.Series, row_index: int) -> dict[str, Any]:
    extra_info = normalize_mapping(row.get("extra_info"))
    interaction_kwargs = normalize_mapping(extra_info.get("interaction_kwargs"))
    award_id = clean_text(extra_info.get("award_id") or row.get("award_id"))
    title = clean_text(
        get_field(row, "award_problem", "awd_titl_txt")
        or extra_info.get("awd_titl_txt")
        or interaction_kwargs.get("problem")
    )
    abstract = clean_text(
        extra_info.get("award_abstract")
        or extra_info.get("awd_abstract_narration")
        or get_field(row, "awd_abstract_narration")
    )
    por_text = extract_por_text(get_field(row, "por") or extra_info.get("por"))
    pi_value = get_field(row, "pi")
    if pi_value is None:
        pi_value = extra_info.get("pi")
    pi_names = extract_pi_names(pi_value)

    if not title:
        raise ValueError(f"Missing title for row {row_index}")
    if not abstract:
        raise ValueError(f"Missing abstract for row {row_index}")

    return {
        "row_index": row_index,
        "sample_id": award_id or f"row_{row_index}",
        "award_id": award_id,
        "title": title,
        "abstract": abstract,
        "por": por_text,
        "pi_names": pi_names,
        "award_eff_date": clean_text(extra_info.get("award_eff_date")),
        "award_exp_date": clean_text(extra_info.get("award_exp_date")),
        "source_row": make_jsonable(row.to_dict()),
        "ground_truth": row.get("ground_truth"),
    }


def load_samples(input_file: Path, start_index: int, max_samples: int | None) -> list[dict[str, Any]]:
    dataframe = pd.read_parquet(input_file)
    samples: list[dict[str, Any]] = []
    for row_index, (_, row) in enumerate(dataframe.iterrows()):
        if row_index < start_index:
            continue
        samples.append(build_sample(row, row_index))
        if max_samples is not None and len(samples) >= max_samples:
            break
    return samples


def ensure_output_file(output_file: Path, overwrite: bool) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if overwrite and output_file.exists():
        output_file.unlink()
    failure_file = output_file.with_name(f"{output_file.stem}_failures.jsonl")
    if overwrite and failure_file.exists():
        failure_file.unlink()


def load_completed_ids(output_file: Path) -> set[str]:
    if not output_file.exists():
        return set()
    completed_ids: set[str] = set()
    with output_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = payload.get("sample_id")
            if sample_id:
                completed_ids.add(str(sample_id))
    return completed_ids


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(make_jsonable(payload), ensure_ascii=False) + "\n")


def append_failure(output_file: Path, payload: dict[str, Any]) -> None:
    failure_file = output_file.with_name(f"{output_file.stem}_failures.jsonl")
    append_jsonl(failure_file, payload)


def tokenize_for_search(text: str, limit: int = 8) -> list[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "at", "award", "based", "by", "career", "collaborative", "for", "from",
        "in", "into", "of", "on", "or", "project", "research", "the", "to", "toward", "towards", "using", "with",
    }
    tokens = [token.lower() for token in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)]
    filtered = [token for token in tokens if token not in stopwords and not token.startswith("nsf")]
    return list(dict.fromkeys(filtered))[:limit]


def last_name(name: str) -> str:
    parts = re.findall(r"[A-Za-z][A-Za-z'-]+", name)
    return parts[-1].lower() if parts else ""


def build_arxiv_queries(sample: dict[str, Any]) -> list[str]:
    pi_names = sample.get("pi_names") or []

    queries: list[str] = []
    cleaned_pi_names = [clean_text(name) for name in pi_names if clean_text(name)]

    # PI-first retrieval: fetch author-matched papers, then let semantic scoring
    # against award abstract/POR decide which are likely connected to this award.
    for pi_name in cleaned_pi_names[:4]:
        quoted_name = pi_name.replace('"', "")
        surname = last_name(pi_name)
        queries.append(f'au:"{quoted_name}"')
        if surname and len(surname) >= 4:
            queries.append(f'au:"{surname}"')

    if len(cleaned_pi_names) >= 2:
        pair = [name.replace('"', "") for name in cleaned_pi_names[:2]]
        queries.append(" OR ".join(f'au:"{name}"' for name in pair))

    if len(cleaned_pi_names) >= 3:
        triple = [name.replace('"', "") for name in cleaned_pi_names[:3]]
        queries.append(" OR ".join(f'au:"{name}"' for name in triple))

    deduped: list[str] = []
    for query in queries:
        if query not in deduped:
            deduped.append(query)
    return deduped[:12]


def fetch_url(url: str, timeout: int = 60) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "creative-rl-synthetic-proposals/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def parse_arxiv_entries(xml_bytes: bytes, query: str) -> list[ArxivPaper]:
    namespace = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(xml_bytes)
    papers: list[ArxivPaper] = []
    for entry in root.findall("atom:entry", namespace):
        entry_id = clean_text(entry.findtext("atom:id", namespaces=namespace)) or ""
        arxiv_id = entry_id.rsplit("/", maxsplit=1)[-1]
        title = clean_text(entry.findtext("atom:title", namespaces=namespace)) or ""
        abstract = clean_text(entry.findtext("atom:summary", namespaces=namespace)) or ""
        authors = [clean_text(author.findtext("atom:name", namespaces=namespace)) or "" for author in entry.findall("atom:author", namespace)]
        pdf_url = ""
        for link in entry.findall("atom:link", namespace):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        papers.append(
            ArxivPaper(
                arxiv_id=arxiv_id,
                title=title,
                abstract=abstract,
                authors=[author for author in authors if author],
                published=clean_text(entry.findtext("atom:published", namespaces=namespace)) or "",
                updated=clean_text(entry.findtext("atom:updated", namespaces=namespace)) or "",
                pdf_url=pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                entry_url=entry_id,
                search_query=query,
            )
        )
    return papers


def search_arxiv(query: str, max_results: int, max_retries: int, backoff_base: float) -> list[ArxivPaper]:
    params = urllib.parse.urlencode(
        {"search_query": query, "start": 0, "max_results": max_results, "sortBy": "relevance", "sortOrder": "descending"}
    )
    url = f"{ARXIV_API_URL}?{params}"

    for attempt in range(max_retries + 1):
        try:
            xml_bytes = fetch_url(url)
            return parse_arxiv_entries(xml_bytes, query)
        except urllib.error.HTTPError as exc:
            retriable = exc.code in {408, 425, 429, 500, 502, 503, 504}
            if not retriable or attempt >= max_retries:
                raise RuntimeError(f"arXiv search failed for {query!r}: HTTP {exc.code}: {exc.reason}") from exc
        except urllib.error.URLError as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"arXiv search failed for {query!r}: {exc}") from exc

        sleep_seconds = backoff_base * (2**attempt) + random.uniform(0.0, 0.75)
        time.sleep(sleep_seconds)

    return []


def year_from_date(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"(19|20)\d{2}", value)
    return int(match.group(0)) if match else None


def paper_metadata_score(paper: ArxivPaper, sample: dict[str, Any]) -> float:
    award_evidence = " ".join(
        [
            sample.get("title") or "",
            sample.get("abstract") or "",
            sample.get("por") or "",
        ]
    )
    award_terms = set(tokenize_for_search(award_evidence, limit=80))
    paper_terms = set(tokenize_for_search(paper.title + " " + paper.abstract, limit=60))
    overlap = len(award_terms & paper_terms)

    pi_last_names = {last_name(name) for name in sample.get("pi_names", []) if last_name(name)}
    author_last_names = {last_name(name) for name in paper.authors if last_name(name)}
    author_overlap = len(pi_last_names & author_last_names)

    published_year = year_from_date(paper.published)
    award_start = year_from_date(sample.get("award_eff_date"))
    award_end = year_from_date(sample.get("award_exp_date"))
    year_bonus = 0.0
    if published_year and award_start:
        end_year = (award_end or award_start + 5) + 2
        if award_start - 1 <= published_year <= end_year:
            year_bonus = 2.0
        elif abs(published_year - award_start) <= 3:
            year_bonus = 1.0

    exact_title_bonus = 2.0 if clean_text(sample["title"]) and clean_text(sample["title"]).lower() in paper.abstract.lower() else 0.0
    return (author_overlap * 4.0) + min(overlap, 12) * 0.7 + year_bonus + exact_title_bonus


def find_candidate_papers(
    sample: dict[str, Any],
    max_results: int,
    arxiv_delay: float,
    arxiv_max_retries: int,
    arxiv_backoff_base: float,
) -> list[ArxivPaper]:
    papers_by_id: dict[str, ArxivPaper] = {}
    for index, query in enumerate(build_arxiv_queries(sample)):
        if index:
            time.sleep(arxiv_delay)
        for paper in search_arxiv(
            query,
            max_results=max_results,
            max_retries=arxiv_max_retries,
            backoff_base=arxiv_backoff_base,
        ):
            paper.metadata_score = paper_metadata_score(paper, sample)
            existing = papers_by_id.get(paper.arxiv_id)
            if existing is None or paper.metadata_score > existing.metadata_score:
                papers_by_id[paper.arxiv_id] = paper
    return sorted(papers_by_id.values(), key=lambda paper: paper.metadata_score, reverse=True)


def run_command(command: list[str], timeout: int = 180) -> str | None:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip() or completed.stderr.strip()
    return text or None


def download_pdf(paper: ArxivPaper, cache_dir: Path) -> Path | None:
    pdf_dir = cache_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path
    try:
        pdf_path.write_bytes(fetch_url(paper.pdf_url, timeout=120))
    except Exception:
        return None
    return pdf_path if pdf_path.stat().st_size > 0 else None


def extract_text_with_arxiv2text(paper: ArxivPaper, pdf_path: Path | None, command_name: str) -> str | None:
    if not shutil.which(command_name):
        return None
    command_variants = [
        [command_name, paper.arxiv_id],
        [command_name, paper.pdf_url],
    ]
    if pdf_path is not None:
        command_variants.append([command_name, str(pdf_path)])
    for command in command_variants:
        text = run_command(command, timeout=240)
        if text and len(text) > 1000:
            return text
    return None


def extract_text_with_pdftotext(pdf_path: Path | None) -> str | None:
    if pdf_path is None or not shutil.which("pdftotext"):
        return None
    output = run_command(["pdftotext", "-layout", str(pdf_path), "-"], timeout=180)
    return output if output and len(output) > 1000 else None


def extract_text_with_python_pdf(pdf_path: Path | None) -> str | None:
    if pdf_path is None:
        return None
    for module_name in ("pypdf", "PyPDF2"):
        try:
            if module_name == "pypdf":
                from pypdf import PdfReader  # type: ignore
            else:
                from PyPDF2 import PdfReader  # type: ignore
        except Exception:
            continue
        try:
            reader = PdfReader(str(pdf_path))
            pages = [(page.extract_text() or "") for page in reader.pages[:18]]
            text = "\n".join(pages)
        except Exception:
            continue
        if len(text) > 1000:
            return text
    return None


def get_paper_text(paper: ArxivPaper, cache_dir: Path, arxiv2text_command: str, keep_failed_text: bool) -> str | None:
    text_dir = cache_dir / "text"
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / f"{paper.arxiv_id.replace('/', '_')}.txt"
    if text_path.exists() and text_path.stat().st_size > 0:
        return text_path.read_text(encoding="utf-8", errors="ignore")

    pdf_path = download_pdf(paper, cache_dir)
    text = (
        extract_text_with_arxiv2text(paper, pdf_path, arxiv2text_command)
        or extract_text_with_pdftotext(pdf_path)
        or extract_text_with_python_pdf(pdf_path)
    )
    if text:
        text_path.write_text(text, encoding="utf-8")
    elif keep_failed_text:
        text_path.write_text("", encoding="utf-8")
    return text


def normalize_paper_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=[a-z])", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def section_excerpt(text: str, headings: tuple[str, ...], next_headings: tuple[str, ...], max_chars: int) -> str:
    heading_pattern = r"(?im)^\s*(?:\d+(?:\.\d+)*\s*)?(" + "|".join(re.escape(item) for item in headings) + r")\s*$"
    match = re.search(heading_pattern, text)
    if not match:
        return ""
    start = match.end()
    next_pattern = r"(?im)^\s*(?:\d+(?:\.\d+)*\s*)?(" + "|".join(re.escape(item) for item in next_headings) + r")\s*$"
    next_match = re.search(next_pattern, text[start:])
    end = start + next_match.start() if next_match else min(len(text), start + max_chars)
    return clean_text(text[start:end])[:max_chars] if clean_text(text[start:end]) else ""


def acknowledgement_excerpt(text: str, award_id: str | None, max_chars: int = 2500) -> str:
    candidates: list[str] = []
    ack = section_excerpt(
        text,
        ("Acknowledgments", "Acknowledgements", "Funding", "Funding Acknowledgments"),
        ("References", "Bibliography", "Appendix", "Supplementary Material"),
        max_chars=max_chars,
    )
    if ack:
        candidates.append(ack)
    if award_id and award_id in text:
        index = text.find(award_id)
        candidates.append(clean_text(text[max(0, index - 900): index + 1600]) or "")
    nsf_match = re.search(r"(?i)(National Science Foundation|\bNSF\b)", text)
    if nsf_match:
        start = max(0, nsf_match.start() - 700)
        candidates.append(clean_text(text[start: nsf_match.start() + 1700]) or "")
    return max(candidates, key=len, default="")[:max_chars]


def processed_paper_context(raw_text: str, text_char_limit: int) -> str:
    text = normalize_paper_text(raw_text)
    abstract = section_excerpt(text, ("Abstract",), ("Introduction", "1 Introduction", "Keywords"), 3000)
    intro = section_excerpt(text, ("Introduction",), ("Background", "Related Work", "Method", "Methods", "Approach", "Model"), 6000)
    methods = section_excerpt(
        text,
        ("Method", "Methods", "Approach", "Methodology", "Model", "Algorithm", "Algorithms"),
        ("Experiment", "Experiments", "Evaluation", "Results", "Discussion", "Conclusion"),
        8000,
    )
    experiments = section_excerpt(
        text,
        ("Experiment", "Experiments", "Evaluation", "Results", "Experimental Results"),
        ("Discussion", "Conclusion", "References", "Bibliography", "Appendix"),
        8000,
    )

    if not any((abstract, intro, methods, experiments)):
        text = re.split(r"(?im)^\s*(References|Bibliography|Appendix)\s*$", text, maxsplit=1)[0]
        return clean_text(text[:text_char_limit]) or ""

    chunks = []
    if abstract:
        chunks.append(f"Abstract: {abstract}")
    if intro:
        chunks.append(f"Introduction: {intro}")
    if methods:
        chunks.append(f"Methods/Approach: {methods}")
    if experiments:
        chunks.append(f"Experiments/Results: {experiments}")
    return "\n\n".join(chunks)[:text_char_limit]


def score_full_text(paper: ArxivPaper, sample: dict[str, Any], raw_text: str) -> tuple[float, str]:
    normalized = raw_text.lower()
    score = 0.0
    award_id = sample.get("award_id")
    if award_id and str(award_id) in normalized:
        score += 15.0
    if "national science foundation" in normalized or re.search(r"\bnsf\b", normalized):
        score += 4.0
    pi_last_names = {last_name(name) for name in sample.get("pi_names", []) if last_name(name)}
    for surname in pi_last_names:
        if surname and re.search(rf"\b{re.escape(surname)}\b", normalized):
            score += 1.5
    award_evidence = " ".join(
        [
            sample.get("title") or "",
            sample.get("abstract") or "",
            sample.get("por") or "",
        ]
    )
    award_terms = set(tokenize_for_search(award_evidence, limit=90))
    text_terms = set(tokenize_for_search(raw_text[:60000], limit=200))
    score += min(len(award_terms & text_terms), 15) * 0.4
    return score, acknowledgement_excerpt(raw_text, award_id)


def select_grounding_papers(
    sample: dict[str, Any],
    candidates: list[ArxivPaper],
    cache_dir: Path,
    arxiv2text_command: str,
    keep_failed_text: bool,
    text_char_limit: int,
    papers_per_award: int,
    min_score: float,
) -> list[ArxivPaper]:
    grounded: list[ArxivPaper] = []
    scored_with_text: list[ArxivPaper] = []
    for paper in candidates[: max(8, papers_per_award * 4)]:
        raw_text = get_paper_text(paper, cache_dir, arxiv2text_command, keep_failed_text)
        if not raw_text:
            continue
        paper.full_text_score, paper.evidence_excerpt = score_full_text(paper, sample, raw_text)
        paper.processed_text = processed_paper_context(raw_text, text_char_limit=text_char_limit)
        if paper.processed_text:
            scored_with_text.append(paper)
        if paper.total_score >= min_score and paper.processed_text:
            grounded.append(paper)
        if len(grounded) >= papers_per_award:
            break

    if grounded:
        return sorted(grounded, key=lambda paper: paper.total_score, reverse=True)[:papers_per_award]

    # If strict threshold rejects everything, still return the strongest PI-matched
    # papers so downstream proposal generation can proceed with semantic grounding.
    return sorted(scored_with_text, key=lambda paper: paper.total_score, reverse=True)[:papers_per_award]


def build_prompt(sample: dict[str, Any], papers: list[ArxivPaper], output_token_target: int = DEFAULT_MAX_TOKENS) -> str:
    pi_text = ", ".join(sample.get("pi_names") or []) or "Unknown"
    por_text = sample.get("por") or "No project outcomes report is available for this sample."
    paper_blocks = []
    for paper_index, paper in enumerate(papers, start=1):
        evidence = paper.evidence_excerpt or "No explicit acknowledgement excerpt found; use this paper as topical and author evidence only."
        paper_blocks.append(
            f"""Paper {paper_index}
Title: {paper.title}
Authors: {", ".join(paper.authors)}
arXiv ID: {paper.arxiv_id}
Published: {paper.published}
Likely-award evidence: {evidence}
Relevant paper content, with references and appendices removed where possible:
{paper.processed_text}
"""
        )
    papers_text = "\n\n".join(paper_blocks) if paper_blocks else "No likely arXiv papers were found."

    return f"""You are an expert NSF principal investigator reconstructing a highly realistic original NSF proposal.

Use the award title, abstract, project outcomes report, PI names, and likely resulting papers to infer a plausible original proposal. The proposal should read like a credible NSF project narrative summary: specific technical hypotheses, clear intellectual merit, realistic broader impacts, grounded methodology, measurable milestones, risks, and fallback plans. Treat the papers as downstream evidence of what the funded project probably produced; do not copy text verbatim from the papers, and do not mention that the proposal is reconstructed or generated.

Award title:
{sample["title"]}

Award ID: {sample.get("award_id") or "Unknown"}
Principal investigator names: {pi_text}
Award period: {sample.get("award_eff_date") or "Unknown"} to {sample.get("award_exp_date") or "Unknown"}

NSF award abstract:
{sample["abstract"]}

NSF project outcomes report:
{por_text}

Likely papers supported by or closely associated with this award:
{papers_text}

Write a detailed proposal that is more realistic and thorough than a generic summary. Infer the pre-award research plan that would plausibly lead to the given papers and outcomes. Make the research plan concrete enough that an evaluator can see the tasks, data, algorithms, analyses, evaluation metrics, expected results, potential pitfalls, and fallback strategies.

Please output JSON only in exactly this schema:
{{
"proposal_title": "string",
"proposal_summary": "string: 2-4 dense paragraphs describing what will be done, why it matters, specific problems resolved, intellectual merit, broader impacts, training/education/outreach when relevant, and why now",
"background_and_significance": "string: detailed historical and technical review of the field, limitations of existing approaches, how this project builds on and differs from prior work, and why the proposed direction is novel and important",
"research_plan": [
    {{
        "phase": "string: name of this research phase",
        "idea": "string: detailed central hypothesis and technical concept for this phase, including expected contribution and how it connects to the award evidence and likely papers",
        "experimental_plan": "string: concrete experimental or research plan, including datasets/materials, algorithms/models/systems/theory, evaluation metrics, analysis methods, timeline-style milestones, expected outcomes, risks, and fallback plans"
    }}
]
}}

Requirements:
- `research_plan` must contain 3 to 5 phases unless the award evidence clearly supports fewer.
- Every phase must contain exactly `phase`, `idea`, and `experimental_plan`.
- The content must remain consistent with the award title, abstract, POR, and likely papers.
- Keep your visible JSON output at or below approximately {output_token_target} tokens.
- Do not include markdown, commentary, citations, or code fences.
- Return valid JSON only.

Your output JSON:
"""


def parse_json_response(content: str) -> Any:
    try:
        return json_repair.loads(content)
    except Exception:
        pass

    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


class AzureProposalClient:
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_version: str,
        reasoning_effort: str,
        reasoning_budget_tokens: int,
    ) -> None:
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=api_version)
        else:
            client_id = os.getenv("CLIENT_ID") or None
            credential = DefaultAzureCredential(managed_identity_client_id=client_id)
            token_provider = get_bearer_token_provider(
                credential,
                "https://cognitiveservices.azure.com/.default",
            )
            self.client = AzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
            )
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.reasoning_budget_tokens = max(0, reasoning_budget_tokens)

    def complete(self, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, dict[str, Any]]:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "response_format": {"type": "json_object"},
        }
        if "gpt-5" in self.model.lower():
            request_kwargs["max_completion_tokens"] = max_tokens + self.reasoning_budget_tokens
            request_kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            request_kwargs["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**request_kwargs)
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""

        usage_obj = getattr(response, "usage", None)
        usage_dict = usage_obj.model_dump() if hasattr(usage_obj, "model_dump") else make_jsonable(usage_obj)
        response_metadata = {
            "response_id": getattr(response, "id", None),
            "response_model": getattr(response, "model", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "refusal": getattr(message, "refusal", None),
            "content_length": len(content),
            "usage": usage_dict,
        }
        return content, response_metadata


def reconstruct_sample(
    client: AzureProposalClient,
    sample: dict[str, Any],
    papers: list[ArxivPaper],
    max_tokens: int,
    max_retries: int = DEFAULT_RETRIES,
) -> tuple[dict[str, Any], str, str, dict[str, Any]]:
    base_prompt = build_prompt(sample, papers, output_token_target=max_tokens)
    messages = [{"role": "user", "content": base_prompt}]
    last_raw_response = ""
    last_response_meta: dict[str, Any] = {}
    last_error = ""

    for attempt in range(max_retries):
        try:
            last_raw_response, last_response_meta = client.complete(messages, max_tokens=max_tokens)
            if not last_raw_response.strip():
                raise ValueError(
                    "Model returned empty content. "
                    f"finish_reason={last_response_meta.get('finish_reason')} "
                    f"refusal={last_response_meta.get('refusal')}"
                )
            parsed = parse_json_response(last_raw_response)
            validated = proposal_schema.model_validate(parsed)
            return validated.model_dump(), last_raw_response, base_prompt, last_response_meta
        except (json.JSONDecodeError, ValidationError, openai.BadRequestError) as exc:
            last_error = str(exc)
            if attempt == max_retries - 1:
                break
            messages = [
                {"role": "user", "content": base_prompt},
                {"role": "assistant", "content": last_raw_response},
                {
                    "role": "user",
                    "content": (
                        "Your previous response did not match the required JSON schema. "
                        f"Validation error: {last_error}. Return corrected JSON only."
                    ),
                },
            ]
        except ValueError as exc:
            last_error = str(exc)
            if attempt == max_retries - 1:
                break
            messages = [
                {"role": "user", "content": base_prompt},
                {
                    "role": "user",
                    "content": (
                        "Your previous response was empty or invalid JSON. "
                        f"Error: {last_error}. Return corrected JSON only."
                    ),
                },
            ]

    raise ProposalGenerationError(
        message=(
            f"Failed to produce valid synthetic proposal JSON for sample {sample['sample_id']}: {last_error}"
        ),
        sample_id=str(sample["sample_id"]),
        last_raw_response=last_raw_response,
        response_metadata=last_response_meta,
    )


def truncate_prompt_if_needed(prompt: str, max_tokens: int) -> str:
    char_budget = max(max_tokens * APPROX_CHARS_PER_TOKEN * 5, 40000)
    if len(prompt) <= char_budget:
        return prompt
    return prompt[:char_budget] + "\n\n[Prompt truncated for dry-run display.]"


def main() -> None:
    args = parse_args()
    endpoint = resolve_endpoint(args.azure_endpoint)
    model = resolve_model(args.model)
    output_file = args.output_file or default_output_path(model)

    ensure_output_file(output_file, overwrite=args.overwrite)
    completed_ids = set() if args.overwrite else load_completed_ids(output_file)
    samples = load_samples(args.input_file, start_index=args.start_index, max_samples=args.max_samples)
    pending_samples = [sample for sample in samples if str(sample["sample_id"]) not in completed_ids]

    print(f"Loaded {len(samples)} samples from {args.input_file}")
    print(f"Writing results to {output_file}")
    print(
        f"Resume status: completed={len(completed_ids)} remaining={len(pending_samples)} "
        f"skipped={len(samples) - len(pending_samples)}"
    )

    client = None
    if not args.dry_run:
        if not endpoint:
            raise ValueError(
                "Azure endpoint not configured. Pass --azure-endpoint or set "
                "SYNTHETIC_PROPOSAL_AZURE_ENDPOINT or CREATIVE_AZURE_ENDPOINTS."
            )
        client = AzureProposalClient(
            endpoint=endpoint,
            model=model,
            api_version=args.api_version,
            reasoning_effort=args.reasoning_effort,
            reasoning_budget_tokens=args.reasoning_budget_tokens,
        )

    failed_samples = 0
    skipped_without_papers = 0

    for sample_count, sample in enumerate(tqdm(pending_samples, desc="Generating synthetic proposals"), start=1):
        try:
            candidates = find_candidate_papers(
                sample,
                max_results=args.paper_search_limit,
                arxiv_delay=args.arxiv_delay,
                arxiv_max_retries=args.arxiv_max_retries,
                arxiv_backoff_base=args.arxiv_backoff_base,
            )
            papers = select_grounding_papers(
                sample=sample,
                candidates=candidates,
                cache_dir=args.paper_cache_dir,
                arxiv2text_command=args.arxiv2text_command,
                keep_failed_text=args.keep_failed_text,
                text_char_limit=args.text_char_limit,
                papers_per_award=args.papers_per_award,
                min_score=args.min_paper_score,
            )
            if not papers and not args.allow_no_papers:
                skipped_without_papers += 1
                append_failure(
                    output_file,
                    {
                        "sample_id": sample["sample_id"],
                        "row_index": sample["row_index"],
                        "award_id": sample["award_id"],
                        "title": sample["title"],
                        "error": "No likely arXiv papers found; skipped.",
                        "candidate_papers": [asdict(paper) for paper in candidates[:10]],
                    },
                )
                continue

            if args.dry_run:
                prompt = build_prompt(sample, papers)
                print(truncate_prompt_if_needed(prompt, args.max_tokens))
                return

            assert client is not None
            proposal_json, raw_response, prompt, response_metadata = reconstruct_sample(
                client=client,
                sample=sample,
                papers=papers,
                max_tokens=args.max_tokens,
            )
            payload = {
                "sample_id": sample["sample_id"],
                "row_index": sample["row_index"],
                "award_id": sample["award_id"],
                "title": sample["title"],
                "abstract": sample["abstract"],
                "por": sample["por"],
                "source_row": sample["source_row"],
                "ground_truth": sample["ground_truth"],
                "generated_proposal": proposal_json,
                "raw_response": raw_response,
                "grounding_papers": [asdict(paper) for paper in papers],
                "candidate_papers": [asdict(paper) for paper in candidates[:10]],
                "generation_prompt": prompt,
                "response_metadata": response_metadata,
            }
            append_jsonl(output_file, payload)
        except Exception as exc:
            failed_samples += 1
            failure_payload = {
                "sample_id": sample["sample_id"],
                "row_index": sample["row_index"],
                "award_id": sample["award_id"],
                "title": sample["title"],
                "error": str(exc),
            }
            if isinstance(exc, ProposalGenerationError):
                failure_payload["response_metadata"] = exc.response_metadata
                failure_payload["raw_response"] = exc.last_raw_response[:2000]
            append_failure(output_file, failure_payload)
            print(f"Skipping failed sample {sample['sample_id']} (row_index={sample['row_index']}): {exc}")
            continue

        if sample_count % args.print_every == 0:
            print(f"Wrote {sample_count} attempted samples to {output_file}")

    print(
        f"Finished. Results written to {output_file}. "
        f"Failed samples: {failed_samples}. Skipped without papers: {skipped_without_papers}."
    )


if __name__ == "__main__":
    main()
