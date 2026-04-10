# mini-search-agent

Stripped-down agent harness for web-search tasks. ~150 lines of scaffolding.

Inspired by [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent): trust the model, minimize the wrapper.

**Model**: Minimax API (function calling)  
**Search**: `minimax-coding-plan-mcp` MCP server — same API key, spawned once per run  
**Loop**: append-only message history; stops when the model returns no tool calls  

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in MINIMAX_API_KEY
export $(cat .env | xargs)

# The MCP server is installed on-demand via uvx (no separate install step)
# Make sure uvx is available: pip install uv
```

---

## Quick test

```bash
python agent.py "Who founded Shiseido and in what year?"
```

---

## Run a benchmark

Tasks CSV must have at minimum: `task_id`, `question` (or `Question`), `answer` (or `Final answer`).

```bash
# GAIA validation set
python bench.py --tasks data/gaia_val.csv --out results/gaia.csv --preset gaia

# WideSearch
python bench.py --tasks data/widesearch.csv --out results/ws.csv --preset widesearch

# Custom system prompt
python bench.py --tasks data/custom.csv --out results/custom.csv \
  --system "You are a precise assistant. Answer concisely."

# Resume an interrupted run (already-completed task_ids are skipped automatically)
python bench.py --tasks data/gaia_val.csv --out results/gaia.csv --preset gaia
```

Results are written to the output CSV incrementally — safe to Ctrl+C and resume.

---

## Presets

| Preset | Use for |
|---|---|
| `gaia` | GAIA Level 1-3 — concise single-answer questions |
| `widesearch` | WideSearch — table cell population from live web |
| `enrichment` | Japanese company name resolution → JSON output |

---

## Files

```
agent.py          Core agent loop + Minimax API + MCP tool execution
bench.py          Benchmark runner with CSV checkpointing and scoring
requirements.txt  mcp, requests
.env.example      Environment variable template
data/             Put your task CSVs here
results/          Output CSVs land here
```

---

## Architecture

```
bench.py
  └─ Agent (agent.py)
       ├─ _call_model()        POST /v1/chat/completions  → Minimax API
       └─ _execute(tool, args)
            ├─ web_search  →  minimax-coding-plan-mcp (MCP subprocess, persistent)
            └─ read_file   →  local filesystem
```

The MCP server is started once when `Agent.__aenter__` is called and shut down when the context exits. All tasks in a benchmark run share the same server process.
