#!/usr/bin/env python3
"""
Benchmark runner — loads tasks, runs agent, saves results with checkpointing.

Usage:
    python -m bench --tasks data/gaia_val.csv --out results/gaia.csv --preset gaia
    python -m bench --out results/ws.jsonl --preset widesearch [--lang en]
    python -m bench --tasks data/custom.csv --out results/custom.csv --system "You are..."
"""

import argparse, asyncio, csv, json, re
from datetime import datetime
from pathlib import Path

import agent as ag

PRESETS = {
    "gaia": (
        "You are a precise research assistant. "
        "Answer the question using web search as needed. "
        "Return a concise final answer — a single word, number, or short phrase. "
        "Do not include explanation unless the question explicitly asks for it."
    ),
    "widesearch": (
        "You are a meticulous data researcher. Your task is to search the web and "
        "compile a comprehensive Markdown table with ALL rows and columns requested. "
        "Completeness is critical — find every item, not just a few examples.\n\n"
        "Rules:\n"
        "- Search systematically. Use multiple queries to cover the full scope.\n"
        "- Return your final answer as a Markdown table inside a ```markdown fence.\n"
        "- Include ALL columns mentioned in the task. Use the exact column names.\n"
        "- Include as many rows as possible — more is better than fewer.\n"
        "- If a cell value is unknown after searching, write 'N/A'.\n"
        "- Do NOT include any text outside the ```markdown fence in your final answer."
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


def load_tasks(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_done(path: str) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {r["task_id"] for r in csv.DictReader(f) if r.get("task_id")}


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def score(predicted: str, expected: str) -> "bool | str":
    if not expected or not expected.strip():
        return ""
    np, ne = _norm(predicted), _norm(expected)
    return np == ne or ne in np


async def run_benchmark(
    tasks_csv: str,
    out_csv: str,
    system_prompt: str,
    delay: float = 0.5,
    verbose: bool = True,
):
    tasks = load_tasks(tasks_csv)
    done  = load_done(out_csv)

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

    print(f"\n{'─'*40}")
    if scored_count > 0:
        print(f"Accuracy : {correct_count/scored_count:.1%}  ({correct_count}/{scored_count})")
    else:
        print("No ground-truth answers — accuracy not calculated.")
    print(f"Output   : {out_csv}")


async def run_widesearch(
    out_path: str,
    system_prompt: str,
    language: str | None = None,
    delay: float = 1.0,
    verbose: bool = True,
    limit: int | None = None,
):
    from bench import widesearch as ws

    tasks = ws.load_tasks(language)
    if limit:
        tasks = tasks[:limit]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    pass

    results: list[ws.EvalResult] = []

    async with ag.Agent() as agent:
        for i, task in enumerate(tasks):
            iid = task["instance_id"]
            if iid in done:
                if verbose:
                    print(f"  [skip] {iid}")
                continue

            query = task["query"]
            if verbose:
                print(f"[{i+1}/{len(tasks)}] {iid}: {query[:80]}{'…' if len(query)>80 else ''}")

            response = ""
            steps = 0
            try:
                r = await agent.run(query, system=system_prompt)
                response = r["answer"]
                steps = r["steps"]
            except Exception as e:
                response = f"ERROR: {e}"

            eval_config = json.loads(task["evaluation"]) if isinstance(task["evaluation"], str) else task["evaluation"]
            try:
                gold_df = ws.load_gold(iid)
                ev = ws.evaluate_task(iid, response, gold_df, eval_config)
            except Exception as e:
                ev = ws.EvalResult(instance_id=iid, error=str(e))

            results.append(ev)

            record = {
                "instance_id": iid,
                "language": task["language"],
                "steps": steps,
                "row_f1": round(ev.row_f1, 4),
                "item_f1": round(ev.item_f1, 4),
                "row_precision": round(ev.row_precision, 4),
                "row_recall": round(ev.row_recall, 4),
                "item_precision": round(ev.item_precision, 4),
                "item_recall": round(ev.item_recall, 4),
                "matched_rows": ev.matched_rows,
                "pred_rows": ev.pred_rows,
                "gold_rows": ev.gold_rows,
                "error": ev.error,
                "response": response[:2000],
                "timestamp": datetime.now().isoformat(),
            }
            with open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            if verbose:
                status = f"row_F1={ev.row_f1:.2f} item_F1={ev.item_f1:.2f}"
                if ev.error:
                    status += f" err={ev.error}"
                print(f"  → {ev.matched_rows}/{ev.gold_rows} rows matched | {status}")

            if delay > 0:
                await asyncio.sleep(delay)

    if results:
        avg_row_f1 = sum(r.row_f1 for r in results) / len(results)
        avg_item_f1 = sum(r.item_f1 for r in results) / len(results)
        errors = sum(1 for r in results if r.error)
        print(f"\n{'─'*50}")
        print(f"WideSearch Results ({len(results)} tasks)")
        print(f"  Avg Row  F1 : {avg_row_f1:.3f}")
        print(f"  Avg Item F1 : {avg_item_f1:.3f}")
        if errors:
            print(f"  Errors      : {errors}")
        print(f"  Output      : {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks",  default=None,              help="Input task CSV (not needed for widesearch)")
    ap.add_argument("--out",    required=True,             help="Output results file")
    ap.add_argument("--preset", default="gaia",
                    choices=list(PRESETS.keys()),           help="System prompt preset")
    ap.add_argument("--system", default=None,              help="Custom system prompt (overrides --preset)")
    ap.add_argument("--delay",  type=float, default=0.5,   help="Seconds between tasks")
    ap.add_argument("--quiet",  action="store_true",       help="Suppress per-task output")
    ap.add_argument("--lang",   default=None, choices=["en", "zh"],
                    help="WideSearch: filter by language")
    ap.add_argument("--limit",  type=int, default=None,    help="WideSearch: max tasks to run")
    args = ap.parse_args()

    system = args.system or PRESETS[args.preset]

    if args.preset == "widesearch":
        asyncio.run(run_widesearch(
            out_path=args.out,
            system_prompt=system,
            language=args.lang,
            delay=args.delay,
            verbose=not args.quiet,
            limit=args.limit,
        ))
    else:
        if not args.tasks:
            ap.error("--tasks is required for non-widesearch presets")
        asyncio.run(run_benchmark(
            tasks_csv=args.tasks,
            out_csv=args.out,
            system_prompt=system,
            delay=args.delay,
            verbose=not args.quiet,
        ))


if __name__ == "__main__":
    main()
