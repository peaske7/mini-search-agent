#!/usr/bin/env python3
"""
Enrichment pipeline — context parsing, prompt building, per-row enrichment, output assembly.
"""

import csv, json, re
from pathlib import Path


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
