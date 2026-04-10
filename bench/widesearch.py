#!/usr/bin/env python3
"""
WideSearch benchmark integration — loader, parser, evaluator.

Dataset : https://huggingface.co/datasets/ByteDance-Seed/WideSearch
Paper   : arXiv 2508.07999

Usage:
    # Download dataset + gold CSVs
    python -m bench.widesearch download

    # Evaluate a results JSONL (instance_id, response)
    python -m bench.widesearch evaluate --results results/ws_run.jsonl
"""

import io, json, os, re, math
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd
import dateparser
import requests

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR  = Path("data/widesearch")
GOLD_DIR  = DATA_DIR / "gold"
TASKS_FILE = DATA_DIR / "tasks.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def download_dataset(language: str | None = None) -> list[dict]:
    """Download WideSearch from HuggingFace and cache locally. Returns tasks."""
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ByteDance-Seed/WideSearch", split="full")

    # Download gold CSVs from the HF repo
    from huggingface_hub import hf_hub_download
    gold_files = [f"widesearch_gold/{r['instance_id']}.csv" for r in ds]
    for gf in gold_files:
        dest = GOLD_DIR / Path(gf).name
        if not dest.exists():
            local = hf_hub_download(
                repo_id="ByteDance-Seed/WideSearch",
                filename=gf,
                repo_type="dataset",
            )
            dest.write_bytes(Path(local).read_bytes())

    # Save tasks as JSONL
    tasks = []
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        for row in ds:
            task = {
                "instance_id": row["instance_id"],
                "query": row["query"],
                "evaluation": row["evaluation"],
                "language": row["language"],
            }
            if language and task["language"] != language:
                continue
            f.write(json.dumps(task, ensure_ascii=False) + "\n")
            tasks.append(task)

    print(f"Saved {len(tasks)} tasks → {TASKS_FILE}")
    print(f"Gold CSVs → {GOLD_DIR}/")
    return tasks


def load_tasks(language: str | None = None) -> list[dict]:
    """Load tasks from local cache. Downloads if missing."""
    if not TASKS_FILE.exists():
        return download_dataset(language)
    tasks = []
    with open(TASKS_FILE, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            if language and t["language"] != language:
                continue
            tasks.append(t)
    return tasks


def load_gold(instance_id: str) -> pd.DataFrame:
    """Load gold-standard CSV for a given instance."""
    path = GOLD_DIR / f"{instance_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Gold CSV not found: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MARKDOWN TABLE PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_markdown_table(text: str) -> pd.DataFrame | None:
    """
    Extract a Markdown pipe-delimited table from agent output.
    Looks for ```markdown fences first, then falls back to bare pipe tables.
    """
    # Try fenced block first
    fenced = re.search(r"```(?:markdown)?\s*\n(.*?)```", text, re.DOTALL)
    block = fenced.group(1).strip() if fenced else text

    # Find pipe-delimited lines
    lines = [l.strip() for l in block.splitlines() if "|" in l]
    if len(lines) < 2:
        return None

    # Remove separator row (e.g., |---|---|)
    data_lines = []
    for line in lines:
        stripped = re.sub(r"[|\s\-:]", "", line)
        if stripped:  # not a pure separator
            data_lines.append(line)

    if len(data_lines) < 2:
        return None

    def split_row(line: str) -> list[str]:
        line = line.strip().strip("|")
        return [cell.strip() for cell in line.split("|")]

    header = split_row(data_lines[0])
    rows = [split_row(l) for l in data_lines[1:]]

    # Pad/truncate rows to match header length
    n = len(header)
    rows = [r[:n] + [""] * max(0, n - len(r)) for r in rows]

    return pd.DataFrame(rows, columns=header)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def norm_str(s: str) -> str:
    """Lowercase, strip whitespace and asterisks."""
    return re.sub(r"[\s*]+", " ", s.lower().strip()).strip()


def norm_column(name: str) -> str:
    """Normalize column name for alignment."""
    return re.sub(r"[\s_\-*]+", "", name.lower().strip())


def extract_number(s: str) -> float | None:
    """Extract first numeric value from string."""
    s = s.replace(",", "").replace("$", "").replace("€", "").replace("¥", "")
    m = re.search(r"-?[\d]+\.?\d*", s)
    return float(m.group()) if m else None


def norm_date(s: str) -> datetime | None:
    """Parse a date string into datetime."""
    if not s or not s.strip():
        return None
    try:
        return dateparser.parse(s, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PER-CELL METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if norm_str(pred) == norm_str(gold) else 0.0


def number_near(pred: str, gold: str, tolerance: float = 0.1) -> float:
    pn, gn = extract_number(pred), extract_number(gold)
    if pn is None or gn is None:
        return exact_match(pred, gold)
    if gn == 0:
        return 1.0 if pn == 0 else 0.0
    return 1.0 if abs(pn - gn) / abs(gn) <= tolerance else 0.0


def url_match(pred: str, gold: str) -> float:
    """Compare URL domains."""
    def domain(u: str) -> str:
        u = re.sub(r"^https?://", "", u.lower().strip().rstrip("/"))
        u = re.sub(r"^www\.", "", u)
        return u.split("/")[0]
    return 1.0 if domain(pred) == domain(gold) else 0.0


def date_near(pred: str, gold: str, max_days: int = 31) -> float:
    dp, dg = norm_date(pred), norm_date(gold)
    if dp is None or dg is None:
        return exact_match(pred, gold)
    return 1.0 if abs((dp - dg).days) <= max_days else 0.0


def in_match(pred: str, gold: str) -> float:
    """Check if normalized pred is contained in gold or vice versa."""
    np, ng = norm_str(pred), norm_str(gold)
    return 1.0 if (np in ng or ng in np) else 0.0


def llm_judge(pred: str, gold: str, criterion: str) -> float:
    """
    Use the Minimax API (same as the agent) to judge semantic equivalence.
    Falls back to exact_match if API is unavailable.
    """
    api_key = os.environ.get("MINIMAX_API_KEY")
    api_host = os.environ.get("MINIMAX_API_HOST", "https://api.minimax.chat")
    model = os.environ.get("EVAL_MODEL", os.environ.get("AGENT_MODEL", "MiniMax-2.7"))

    if not api_key:
        return exact_match(pred, gold)

    prompt = (
        f"Judge whether the predicted value matches the gold value.\n"
        f"Criterion: {criterion}\n"
        f"Gold: {gold}\n"
        f"Predicted: {pred}\n\n"
        f"Reply with ONLY a number: 1.0 if it matches, 0.0 if not."
    )
    try:
        resp = requests.post(
            f"{api_host}/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        score = float(re.search(r"[01]\.?\d*", text).group())
        return min(max(score, 0.0), 1.0)
    except Exception:
        return exact_match(pred, gold)


METRICS = {
    "exact_match": exact_match,
    "number_near": number_near,
    "url_match":   url_match,
    "date_near":   date_near,
    "in_match":    in_match,
    "llm_judge":   llm_judge,
}


def score_cell(pred: str, gold: str, pipeline: dict) -> float:
    """Score a single cell using its eval pipeline config."""
    # Apply preprocessing
    for pp in pipeline.get("preprocess", []):
        if pp == "norm_str":
            pred, gold = norm_str(pred), norm_str(gold)
        elif pp == "extract_number":
            pn, gn = extract_number(pred), extract_number(gold)
            if pn is not None:
                pred = str(pn)
            if gn is not None:
                gold = str(gn)
        elif pp == "norm_date":
            dp, dg = norm_date(pred), norm_date(gold)
            if dp:
                pred = dp.strftime("%Y-%m-%d")
            if dg:
                gold = dg.strftime("%Y-%m-%d")

    # Apply metrics (take max across all specified metrics)
    metrics = pipeline.get("metric", ["exact_match"])
    criterion = pipeline.get("criterion", "")

    best = 0.0
    for m in metrics:
        fn = METRICS.get(m, exact_match)
        if m == "number_near" and isinstance(criterion, (int, float)):
            best = max(best, fn(pred, gold, float(criterion)))
        elif m == "llm_judge" and criterion:
            best = max(best, fn(pred, gold, str(criterion)))
        else:
            best = max(best, fn(pred, gold))
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# 5. COLUMN & ROW ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def align_columns(pred_cols: list[str], required: list[str]) -> dict[str, str]:
    """
    Map predicted column names to required (gold) column names.
    Returns {pred_col: gold_col} mapping.
    """
    mapping = {}
    norm_required = {norm_column(r): r for r in required}
    unmatched_gold = dict(norm_required)

    for pc in pred_cols:
        nc = norm_column(pc)
        if nc in unmatched_gold:
            mapping[pc] = unmatched_gold.pop(nc)

    # Second pass: fuzzy substring matching for remaining
    for pc in pred_cols:
        if pc in mapping:
            continue
        nc = norm_column(pc)
        for ng, orig in list(unmatched_gold.items()):
            if nc in ng or ng in nc:
                mapping[pc] = orig
                del unmatched_gold[ng]
                break

    return mapping


def match_rows(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    unique_cols: list[str],
    col_map: dict[str, str],
) -> list[tuple[int, int]]:
    """
    Match predicted rows to gold rows via primary key columns.
    Returns list of (pred_idx, gold_idx) pairs.
    """
    # Reverse map: gold_col -> pred_col
    rev_map = {v: k for k, v in col_map.items()}
    pred_key_cols = [rev_map[uc] for uc in unique_cols if uc in rev_map]
    gold_key_cols = [uc for uc in unique_cols if uc in rev_map]

    if not pred_key_cols:
        # No key columns found — match by position
        return [(i, i) for i in range(min(len(pred_df), len(gold_df)))]

    # Build key → index maps
    def make_key(df: pd.DataFrame, cols: list[str], idx: int) -> str:
        return "||".join(norm_str(str(df.iloc[idx][c])) for c in cols)

    gold_keys = {}
    for gi in range(len(gold_df)):
        k = make_key(gold_df, gold_key_cols, gi)
        gold_keys[k] = gi

    matches = []
    for pi in range(len(pred_df)):
        k = make_key(pred_df, pred_key_cols, pi)
        if k in gold_keys:
            matches.append((pi, gold_keys.pop(k)))

    return matches


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TASK-LEVEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    instance_id: str
    row_precision: float = 0.0
    row_recall: float = 0.0
    row_f1: float = 0.0
    item_precision: float = 0.0
    item_recall: float = 0.0
    item_f1: float = 0.0
    matched_rows: int = 0
    pred_rows: int = 0
    gold_rows: int = 0
    error: str = ""


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def evaluate_task(
    instance_id: str,
    agent_output: str,
    gold_df: pd.DataFrame,
    eval_config: dict,
) -> EvalResult:
    """Evaluate a single WideSearch task."""
    result = EvalResult(instance_id=instance_id, gold_rows=len(gold_df))

    # Parse predicted table
    pred_df = parse_markdown_table(agent_output)
    if pred_df is None or pred_df.empty:
        result.error = "no_table_parsed"
        return result

    result.pred_rows = len(pred_df)

    # Normalize gold columns to match eval_config's normalized names.
    # eval_config uses normalized keys (e.g. "brand"), gold CSV has originals ("Brand").
    required = eval_config.get("required", [norm_column(c) for c in gold_df.columns])
    gold_col_map = align_columns(list(gold_df.columns), required)
    gold_df = gold_df.rename(columns=gold_col_map)

    # Align predicted columns to the same normalized names
    col_map = align_columns(list(pred_df.columns), required)
    if not col_map:
        result.error = "no_columns_matched"
        return result

    # Row matching (now both DFs and unique_cols use the same normalized namespace)
    unique_cols = eval_config.get("unique_columns", [])
    matches = match_rows(pred_df, gold_df, unique_cols, col_map)
    result.matched_rows = len(matches)

    if not matches:
        result.error = "no_rows_matched"
        return result

    # Per-cell scoring
    pipeline = eval_config.get("eval_pipeline", {})
    row_scores = []  # min score per row (for row-level metric)
    cell_scores = []  # all individual cell scores (for item-level metric)

    eval_cols = [c for c in required if c in set(col_map.values())]
    rev_map = {v: k for k, v in col_map.items()}

    for pi, gi in matches:
        row_min = 1.0
        for gold_col in eval_cols:
            pred_col = rev_map.get(gold_col)
            if pred_col is None:
                cell_scores.append(0.0)
                row_min = 0.0
                continue

            pred_val = str(pred_df.iloc[pi].get(pred_col, ""))
            gold_val = str(gold_df.iloc[gi].get(gold_col, ""))

            col_pipeline = pipeline.get(gold_col, {"metric": ["exact_match"]})
            s = score_cell(pred_val, gold_val, col_pipeline)
            cell_scores.append(s)
            row_min = min(row_min, s)

        row_scores.append(row_min)

    # Aggregate — Row-level P/R/F1
    tp_row = sum(row_scores)
    result.row_precision = tp_row / result.pred_rows if result.pred_rows > 0 else 0.0
    result.row_recall = tp_row / result.gold_rows if result.gold_rows > 0 else 0.0
    result.row_f1 = _f1(result.row_precision, result.row_recall)

    # Aggregate — Item-level P/R/F1
    tp_item = sum(cell_scores)
    total_pred_cells = result.pred_rows * len(eval_cols)
    total_gold_cells = result.gold_rows * len(eval_cols)
    result.item_precision = tp_item / total_pred_cells if total_pred_cells > 0 else 0.0
    result.item_recall = tp_item / total_gold_cells if total_gold_cells > 0 else 0.0
    result.item_f1 = _f1(result.item_precision, result.item_recall)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "download"

    if cmd == "download":
        lang = sys.argv[2] if len(sys.argv) > 2 else None
        tasks = download_dataset(lang)
        print(f"\nDownloaded {len(tasks)} tasks.")
    else:
        print(f"Usage: python -m bench.widesearch download [en|zh]")
