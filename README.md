# mini-search-agent

Minimal agent harness that <120 lines for web-search tasks. Trust the model, minimize the wrapper.

**Model**: Minimax API (function calling)  
**Search**: `minimax-coding-plan-mcp` MCP server (same API key, spawned once per run)  
**Tools**: `web_search`, `bash`  
**Loop**: append-only message history; stops when the model returns no tool calls

## Getting started

```bash
uv sync
cp .env.example .env   # fill in MINIMAX_API_KEY
python agent.py "Who founded Shiseido and in what year?"
```

## Benchmarks

### GAIA (single-answer)

```bash
python -m bench --tasks data/gaia_val.csv --out results/gaia.csv --preset gaia
```

### WideSearch (table collection)

Downloads 200 tasks from HuggingFace automatically. The agent searches the web and returns a Markdown table; evaluation scores each cell with metrics like exact match, number proximity, URL domain match, date proximity, and LLM-as-judge.

```bash
# Download dataset
python -m bench.widesearch download

# Run 10 English tasks
python -m bench --out results/ws.jsonl --preset widesearch --lang en --limit 10

# Run all English tasks
python -m bench --out results/ws.jsonl --preset widesearch --lang en

# Resume an interrupted run (completed tasks are skipped)
python -m bench --out results/ws.jsonl --preset widesearch --lang en
```

### Custom

```bash
python -m bench --tasks data/custom.csv --out results/custom.csv \
  --system "You are a precise assistant. Answer concisely."
```

### Enrichment (company scoring)

Enrich and score companies from a CSV against user-defined criteria.

```bash
# With context file (criteria, instructions, examples)
python -m bench --preset enrich --tasks data/companies.csv \
  --out results/leads.csv --context data/criteria.md

# With inline criteria
python -m bench --preset enrich --tasks data/companies.csv \
  --out results/leads.csv --criteria "B2B cosmetics brands with DTC in Japan"
```

Outputs two files:

- `results/leads.csv` — all rows, enriched with score/qualifies/reasoning
- `results/leads_qualified.csv` — only qualifying rows

See `data/test_criteria.md` for context file format.

## Presets

| Preset | Use for |
| --- | --- |
| `gaia` | GAIA Level 1-3, concise single-answer questions |
| `widesearch` | WideSearch, structured table collection from live web |
| `enrichment` | Japanese company name resolution, JSON output |
| `enrich` | Company enrichment & scoring against user-defined criteria |
