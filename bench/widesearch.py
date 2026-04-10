#!/usr/bin/env python3
"""
WideSearch benchmark — loader, Markdown table parser, multi-metric evaluator.
https://huggingface.co/datasets/ByteDance-Seed/WideSearch
"""

import json, os, re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

import pandas as pd
import dateparser
import requests

DATA_DIR   = Path("data/widesearch")
GOLD_DIR   = DATA_DIR / "gold"
TASKS_FILE = DATA_DIR / "tasks.jsonl"


def download_dataset(language: str | None = None) -> list[dict]:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("ByteDance-Seed/WideSearch", split="full")

    for r in ds:
        dest = GOLD_DIR / f"{r['instance_id']}.csv"
        if not dest.exists():
            local = hf_hub_download(
                repo_id="ByteDance-Seed/WideSearch",
                filename=f"widesearch_gold/{r['instance_id']}.csv",
                repo_type="dataset",
            )
            dest.write_bytes(Path(local).read_bytes())

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
    path = GOLD_DIR / f"{instance_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Gold CSV not found: {path}")
    return pd.read_csv(path, dtype=str).fillna("")


def parse_markdown_table(text: str) -> pd.DataFrame | None:
    fenced = re.search(r"```(?:markdown)?\s*\n(.*?)```", text, re.DOTALL)
    block = fenced.group(1).strip() if fenced else text

    lines = [l.strip() for l in block.splitlines() if "|" in l]
    if len(lines) < 2:
        return None

    data_lines = []
    for line in lines:
        if re.sub(r"[|\s\-:]", "", line):
            data_lines.append(line)

    if len(data_lines) < 2:
        return None

    def split_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    header = split_row(data_lines[0])
    rows = [split_row(l) for l in data_lines[1:]]
    n = len(header)
    rows = [r[:n] + [""] * max(0, n - len(r)) for r in rows]
    return pd.DataFrame(rows, columns=header)


def norm_str(s: str) -> str:
    return re.sub(r"[\s*]+", " ", s.lower().strip()).strip()


def norm_column(name: str) -> str:
    return re.sub(r"[\s_\-*]+", "", name.lower().strip())


def extract_number(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").replace("€", "").replace("¥", "")
    m = re.search(r"-?[\d]+\.?\d*", s)
    return float(m.group()) if m else None


def norm_date(s: str) -> datetime | None:
    if not s or not s.strip():
        return None
    try:
        return dateparser.parse(s, settings={"PREFER_DAY_OF_MONTH": "first"})
    except Exception:
        return None


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
    np, ng = norm_str(pred), norm_str(gold)
    return 1.0 if (np in ng or ng in np) else 0.0


def llm_judge(pred: str, gold: str, criterion: str) -> float:
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


def align_columns(pred_cols: list[str], required: list[str]) -> dict[str, str]:
    mapping = {}
    norm_required = {norm_column(r): r for r in required}
    unmatched = dict(norm_required)

    for pc in pred_cols:
        nc = norm_column(pc)
        if nc in unmatched:
            mapping[pc] = unmatched.pop(nc)

    for pc in pred_cols:
        if pc in mapping:
            continue
        nc = norm_column(pc)
        for ng, orig in list(unmatched.items()):
            if nc in ng or ng in nc:
                mapping[pc] = orig
                del unmatched[ng]
                break

    return mapping


def match_rows(
    pred_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    unique_cols: list[str],
    col_map: dict[str, str],
) -> list[tuple[int, int]]:
    rev_map = {v: k for k, v in col_map.items()}
    pred_key_cols = [rev_map[uc] for uc in unique_cols if uc in rev_map]
    gold_key_cols = [uc for uc in unique_cols if uc in rev_map]

    if not pred_key_cols:
        return [(i, i) for i in range(min(len(pred_df), len(gold_df)))]

    def make_key(df: pd.DataFrame, cols: list[str], idx: int) -> str:
        return "||".join(norm_str(str(df.iloc[idx][c])) for c in cols)

    gold_keys = {}
    for gi in range(len(gold_df)):
        gold_keys[make_key(gold_df, gold_key_cols, gi)] = gi

    matches = []
    for pi in range(len(pred_df)):
        k = make_key(pred_df, pred_key_cols, pi)
        if k in gold_keys:
            matches.append((pi, gold_keys.pop(k)))

    return matches


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
    result = EvalResult(instance_id=instance_id, gold_rows=len(gold_df))

    pred_df = parse_markdown_table(agent_output)
    if pred_df is None or pred_df.empty:
        result.error = "no_table_parsed"
        return result

    result.pred_rows = len(pred_df)

    # eval_config uses normalized keys ("brand"), gold CSV has originals ("Brand")
    required = eval_config.get("required", [norm_column(c) for c in gold_df.columns])
    gold_col_map = align_columns(list(gold_df.columns), required)
    gold_df = gold_df.rename(columns=gold_col_map)

    col_map = align_columns(list(pred_df.columns), required)
    if not col_map:
        result.error = "no_columns_matched"
        return result

    unique_cols = eval_config.get("unique_columns", [])
    matches = match_rows(pred_df, gold_df, unique_cols, col_map)
    result.matched_rows = len(matches)

    if not matches:
        result.error = "no_rows_matched"
        return result

    pipeline = eval_config.get("eval_pipeline", {})
    row_scores = []
    cell_scores = []

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

    tp_row = sum(row_scores)
    result.row_precision = tp_row / result.pred_rows if result.pred_rows > 0 else 0.0
    result.row_recall = tp_row / result.gold_rows if result.gold_rows > 0 else 0.0
    result.row_f1 = _f1(result.row_precision, result.row_recall)

    tp_item = sum(cell_scores)
    total_pred_cells = result.pred_rows * len(eval_cols)
    total_gold_cells = result.gold_rows * len(eval_cols)
    result.item_precision = tp_item / total_pred_cells if total_pred_cells > 0 else 0.0
    result.item_recall = tp_item / total_gold_cells if total_gold_cells > 0 else 0.0
    result.item_f1 = _f1(result.item_precision, result.item_recall)

    return result


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "download"

    if cmd == "download":
        lang = sys.argv[2] if len(sys.argv) > 2 else None
        tasks = download_dataset(lang)
        print(f"\nDownloaded {len(tasks)} tasks.")
    else:
        print(f"Usage: python -m bench.widesearch download [en|zh]")
