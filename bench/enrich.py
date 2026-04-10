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
