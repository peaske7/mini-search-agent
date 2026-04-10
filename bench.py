#!/usr/bin/env python3
"""
Benchmark runner — loads tasks from CSV, runs agent, saves results.
Resumes automatically from checkpoint (already-completed task_ids are skipped).

Usage:
    python bench.py --tasks data/gaia_val.csv --out results/gaia.csv --preset gaia
    python bench.py --tasks data/widesearch.csv --out results/ws.csv --preset widesearch
    python bench.py --tasks data/custom.csv --out results/custom.csv --system "You are..."

Expected CSV columns:
    GAIA      : task_id, Question, Final answer, Level
    WideSearch: task_id, question, answer          (answer may be empty)
    Custom    : task_id, question, answer          (answer may be empty)
"""

import argparse, asyncio, csv, re, time
from datetime import datetime
from pathlib import Path

import agent as ag

# ── System prompt presets ─────────────────────────────────────────────────────
PRESETS = {
    "gaia": (
        "You are a precise research assistant. "
        "Answer the question using web search as needed. "
        "Return a concise final answer — a single word, number, or short phrase. "
        "Do not include explanation unless the question explicitly asks for it."
    ),
    "widesearch": (
        "You are a data researcher populating a table. "
        "For each question, search the web to find the most accurate current value. "
        "Return only the answer value, nothing else."
    ),
    "enrichment": (
        "You are a Japanese business analyst. "
        "Given a company name string (possibly messy), use web search to determine: "
        "1) the canonical company/brand name, "
        "2) whether it is a cosmetics/beauty brand (yes/no/unclear), "
        "3) the parent group if known. "
        'Reply ONLY as JSON: {"canonical":"...","is_cosmetics":"yes|no|unclear",'
        '"parent":"...","category":"brand|temp_agency|salon|retailer|other","note":"..."}'
    ),
}

# ── CSV helpers ───────────────────────────────────────────────────────────────
def load_tasks(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_done(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["task_id"] for r in csv.DictReader(f) if r.get("task_id")}


# ── Scoring ───────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def score(predicted: str, expected: str) -> "bool | str":
    if not expected or not expected.strip():
        return ""  # no ground truth — leave blank
    return _norm(predicted) == _norm(expected)


# ── Main runner ───────────────────────────────────────────────────────────────
async def run_benchmark(
    tasks_csv: str,
    out_csv: str,
    system_prompt: str,
    delay: float = 0.5,
    verbose: bool = True,
):
    tasks = load_tasks(tasks_csv)
    done  = load_done(out_csv)

    # Detect GAIA vs generic column names
    sample = tasks[0] if tasks else {}
    q_col  = "Question"     if "Question"     in sample else "question"
    a_col  = "Final answer" if "Final answer" in sample else "answer"
    id_col = next((k for k in ("task_id", "id") if k in sample), None)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists() or out_path.stat().st_size == 0

    fieldnames    = ["task_id", "question", "expected", "predicted", "correct", "steps", "timestamp"]
    correct_count = 0
    scored_count  = 0

    async with ag.Agent() as agent:
        with open(out_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for i, task in enumerate(tasks):
                tid      = (task.get(id_col) if id_col else None) or str(i)
                question = task.get(q_col, "").strip()
                expected = task.get(a_col, "").strip()

                if tid in done:
                    if verbose:
                        print(f"  [skip] {tid}")
                    continue
                if not question:
                    continue

                if verbose:
                    print(f"[{i+1}/{len(tasks)}] {tid}: {question[:70]}{'...' if len(question)>70 else ''}")

                predicted, steps = "", 0
                try:
                    result    = await agent.run(question, system=system_prompt)
                    predicted = result["answer"].strip()
                    steps     = result["steps"]
                except Exception as e:
                    predicted = f"ERROR: {e}"

                is_correct = score(predicted, expected)
                if isinstance(is_correct, bool):
                    scored_count += 1
                    if is_correct:
                        correct_count += 1

                if verbose:
                    status = "✓" if is_correct is True else ("✗" if is_correct is False else "?")
                    print(f"  → {predicted[:80]} {status}")

                writer.writerow({
                    "task_id":   tid,
                    "question":  question,
                    "expected":  expected,
                    "predicted": predicted,
                    "correct":   is_correct,
                    "steps":     steps,
                    "timestamp": datetime.now().isoformat(),
                })
                f.flush()

                if delay > 0:
                    await asyncio.sleep(delay)

    # Summary
    print(f"\n{'─'*40}")
    if scored_count > 0:
        print(f"Accuracy : {correct_count/scored_count:.1%}  ({correct_count}/{scored_count})")
    else:
        print("No ground-truth answers — accuracy not calculated.")
    print(f"Output   : {out_csv}")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks",  required=True,             help="Input task CSV")
    ap.add_argument("--out",    required=True,             help="Output results CSV")
    ap.add_argument("--preset", default="gaia",
                    choices=list(PRESETS.keys()),           help="System prompt preset")
    ap.add_argument("--system", default=None,              help="Custom system prompt (overrides --preset)")
    ap.add_argument("--delay",  type=float, default=0.5,   help="Seconds between tasks")
    ap.add_argument("--quiet",  action="store_true",       help="Suppress per-task output")
    args = ap.parse_args()

    system = args.system or PRESETS[args.preset]
    asyncio.run(run_benchmark(
        tasks_csv=args.tasks,
        out_csv=args.out,
        system_prompt=system,
        delay=args.delay,
        verbose=not args.quiet,
    ))


if __name__ == "__main__":
    main()
