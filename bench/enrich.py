#!/usr/bin/env python3
"""
Enrichment pipeline — context parsing, prompt building, per-row enrichment, output assembly.
"""

import asyncio, csv, json, re
from datetime import datetime
from pathlib import Path


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from agent response text."""
    # Try fenced code block first
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try raw JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def parse_context(path: str) -> dict:
    """Parse a Markdown context file into sections.

    Returns dict with keys: criteria, fields, instructions, examples_fit, examples_nofit.
    All values are strings (raw section content). Missing sections are empty strings.
    """
    text = Path(path).read_text(encoding="utf-8")
    sections = {}
    current = None
    lines = []

    for line in text.splitlines():
        m = re.match(r"^#\s+(.+)", line)
        if m:
            if current is not None:
                sections[current] = "\n".join(lines).strip()
            current = m.group(1).strip().lower()
            lines = []
        else:
            lines.append(line)

    if current is not None:
        sections[current] = "\n".join(lines).strip()

    return {
        "criteria": sections.get("criteria", ""),
        "fields": sections.get("fields", ""),
        "instructions": sections.get("instructions", ""),
        "examples_fit": sections.get("examples: fit", ""),
        "examples_nofit": sections.get("examples: no fit", ""),
    }


DEFAULT_FIELDS = """- canonical_name: Official company/brand name
- industry: Primary business category
- website: Company homepage URL
- description: One-line company summary
- parent_company: Parent/holding company if applicable
- headquarters: HQ location"""


def build_system_prompt(
    criteria: str = "",
    fields: str = "",
    instructions: str = "",
    examples_fit: str = "",
    examples_nofit: str = "",
    labeled_rows: list[dict] | None = None,
) -> str:
    """Assemble the enrichment system prompt from context sections."""
    parts = [
        "You are a company research analyst. "
        "Given a company name (and possibly additional context), "
        "use web search and bash to research the company and fill in the requested fields. "
        "Then score it against the user's criteria.",
    ]

    # Fields
    f = fields or DEFAULT_FIELDS
    parts.append(f"\n## Fields to research\n{f}")

    # Criteria
    if criteria:
        parts.append(f"\n## Scoring criteria\n{criteria}")

    # Instructions
    if instructions:
        parts.append(f"\n## Research instructions\n{instructions}")

    # Examples
    examples_parts = []
    if examples_fit:
        examples_parts.append(f"### Companies that fit\n{examples_fit}")
    if examples_nofit:
        examples_parts.append(f"### Companies that do NOT fit\n{examples_nofit}")
    if labeled_rows:
        fit = [r for r in labeled_rows if r.get("label", r.get("_label", "")).lower() in ("fit", "positive")]
        nofit = [r for r in labeled_rows if r.get("label", r.get("_label", "")).lower() in ("no_fit", "negative")]
        if fit:
            examples_parts.append("### Labeled examples (fit)\n" + "\n".join(
                f"- {r.get('company', r.get('name', str(r)))}" for r in fit
            ))
        if nofit:
            examples_parts.append("### Labeled examples (no fit)\n" + "\n".join(
                f"- {r.get('company', r.get('name', str(r)))}" for r in nofit
            ))
    if examples_parts:
        parts.append("\n## Examples\n" + "\n\n".join(examples_parts))

    # Output format
    parts.append(
        "\n## Output format\n"
        "Reply ONLY with a single JSON object. It MUST include these keys:\n"
        '- "score": integer 0-100 (how well the company matches the criteria)\n'
        '- "qualifies": true or false\n'
        '- "reasoning": brief explanation of the score\n'
        "Plus all the fields listed above.\n"
        "Do NOT include any text outside the JSON."
    )

    return "\n".join(parts)


async def run_enrichment(
    tasks_csv: str,
    out_path: str,
    criteria: str = "",
    context_path: str | None = None,
    delay: float = 1.0,
    verbose: bool = True,
):
    """Run per-row enrichment and scoring. Writes JSONL checkpoint, then assembles CSVs."""
    import agent as ag

    # Load context
    ctx = parse_context(context_path) if context_path else {}
    # CLI criteria is fallback; context file criteria wins
    effective_criteria = ctx.get("criteria") or criteria

    # Load input CSV
    with open(tasks_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No rows in input CSV.")
        return

    # Detect label column
    label_col = next((c for c in ("label", "_label") if c in rows[0]), None)

    # Split labeled examples vs. rows to enrich
    labeled_rows = []
    enrich_rows = []
    for i, row in enumerate(rows):
        row["_row_id"] = row.get("task_id", row.get("id", str(i)))
        if label_col and row.get(label_col, "").strip():
            labeled_rows.append(row)
        else:
            enrich_rows.append(row)

    # Build system prompt
    system = build_system_prompt(
        criteria=effective_criteria,
        fields=ctx.get("fields", ""),
        instructions=ctx.get("instructions", ""),
        examples_fit=ctx.get("examples_fit", ""),
        examples_nofit=ctx.get("examples_nofit", ""),
        labeled_rows=labeled_rows,
    )

    # JSONL checkpoint path
    jsonl_path = Path(out_path).with_suffix(".jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-done IDs
    done: set[str] = set()
    results: list[dict] = []
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done.add(rec["_row_id"])
                    results.append(rec)
                except (json.JSONDecodeError, KeyError):
                    pass

    if verbose and done:
        print(f"Resuming: {len(done)} rows already completed.")

    # Run enrichment
    async with ag.Agent() as agent:
        for i, row in enumerate(enrich_rows):
            rid = row["_row_id"]
            if rid in done:
                if verbose:
                    print(f"  [skip] {rid}")
                continue

            # Build per-row user prompt from all input columns
            row_data = "\n".join(f"- {k}: {v}" for k, v in row.items() if k != "_row_id" and v)
            user_prompt = f"Research this company and score it:\n\n{row_data}"

            if verbose:
                # Find a display name from common column names
                name = row.get("company", row.get("name", row.get("canonical_name",
                       row.get("brands", row.get("facility_name", rid)))))
                print(f"[{i+1}/{len(enrich_rows)}] {rid}: {str(name)[:70]}")

            enriched = {"_row_id": rid}
            # Preserve original columns
            for k, v in row.items():
                if k != "_row_id":
                    enriched[f"_orig_{k}"] = v

            try:
                result = await agent.run(user_prompt, system=system, verbose=verbose)
                answer = result["answer"]
                enriched["_steps"] = result["steps"]

                parsed = _extract_json(answer)
                if parsed:
                    enriched.update(parsed)
                else:
                    enriched["_raw_response"] = answer[:2000]
                    enriched["score"] = -1
                    enriched["qualifies"] = False
                    enriched["reasoning"] = "parse_error"

            except Exception as e:
                enriched["score"] = -1
                enriched["qualifies"] = False
                enriched["reasoning"] = f"error: {e}"

            enriched["_timestamp"] = datetime.now().isoformat()
            results.append(enriched)
            done.add(rid)

            # Append to JSONL checkpoint
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

            if verbose:
                score = enriched.get("score", "?")
                qual = enriched.get("qualifies", "?")
                print(f"  → score={score} qualifies={qual}")

            if delay > 0:
                await asyncio.sleep(delay)

    # Assemble final CSV
    _write_csv(results, out_path, verbose)

    # Write filtered CSV
    qualified = [r for r in results if r.get("qualifies") is True or str(r.get("qualifies", "")).lower() == "true"]
    filtered_path = str(Path(out_path).with_name(Path(out_path).stem + "_qualified" + Path(out_path).suffix))
    _write_csv(qualified, filtered_path, verbose, label="Qualified")


def _write_csv(records: list[dict], path: str, verbose: bool, label: str = "Enriched"):
    """Write records to CSV. Headers are the union of all keys across records."""
    if not records:
        if verbose:
            print(f"{label}: no records to write.")
        return

    # Collect all keys, put _row_id first, score/qualifies/reasoning early
    all_keys: list[str] = []
    seen: set[str] = set()
    priority = ["_row_id", "score", "qualifies", "reasoning"]
    for k in priority:
        if any(k in r for r in records):
            all_keys.append(k)
            seen.add(k)
    for r in records:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    if verbose:
        print(f"{label}: {len(records)} rows → {path}")
