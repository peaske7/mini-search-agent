#!/usr/bin/env python3
"""
Stripped-down agent harness. ~150 lines. Inspired by mini-swe-agent.

Model   : Minimax API  (MINIMAX_API_KEY)
Search  : minimax-coding-plan-mcp  (same key, spawned once as subprocess)
Tools   : web_search, read_file
Loop    : append-only message history; stop when model returns no tool_calls
"""

import asyncio, json, os
from contextlib import AsyncExitStack
from pathlib import Path

import requests
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── Config (env) ─────────────────────────────────────────────────────────────
MINIMAX_KEY  = os.environ["MINIMAX_API_KEY"]
API_HOST     = os.environ.get("MINIMAX_API_HOST", "https://api.minimax.chat")
MODEL        = os.environ.get("AGENT_MODEL", "MiniMax-Text-01")
MAX_STEPS    = int(os.environ.get("AGENT_MAX_STEPS", "12"))

# ── Tool schemas (sent to model) ──────────────────────────────────────────────
TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web. Returns titles, URLs, and snippets. "
                "Use for current facts, company lookups, verification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a local text, CSV, or JSON file (first 4000 chars).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"],
            },
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────
def _read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"[read_file] File not found: {path}"
    return p.read_text(encoding="utf-8", errors="replace")[:4000]


# ── Minimax API ───────────────────────────────────────────────────────────────
def _call_model(messages: list) -> dict:
    resp = requests.post(
        f"{API_HOST}/v1/chat/completions",
        headers={"Authorization": f"Bearer {MINIMAX_KEY}",
                 "Content-Type": "application/json"},
        json={"model": MODEL, "messages": messages,
              "tools": TOOL_DEFS, "tool_choice": "auto"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]


# ── Agent ─────────────────────────────────────────────────────────────────────
class Agent:
    """
    Single-session agent. Keeps the MCP server alive for the entire run.

    Usage:
        async with Agent() as agent:
            result = await agent.run("What is X?")
            print(result["answer"])
    """

    def __init__(self):
        self._mcp: ClientSession | None = None
        self._stack: AsyncExitStack | None = None

    async def __aenter__(self):
        self._stack = AsyncExitStack()
        server = StdioServerParameters(
            command="uvx",
            args=["minimax-coding-plan-mcp"],
            env={**os.environ,
                 "MINIMAX_API_KEY":  MINIMAX_KEY,
                 "MINIMAX_API_HOST": API_HOST},
        )
        read, write = await self._stack.enter_async_context(stdio_client(server))
        self._mcp   = await self._stack.enter_async_context(ClientSession(read, write))
        await self._mcp.initialize()
        return self

    async def __aexit__(self, *_):
        if self._stack:
            await self._stack.aclose()

    # ── tool execution ────────────────────────────────────────────────────────
    async def _execute(self, name: str, args: dict) -> str:
        if name == "web_search":
            result = await self._mcp.call_tool("web_search", {"query": args.get("query", "")})
            return "\n".join(c.text for c in result.content if hasattr(c, "text")) or "No results."
        if name == "read_file":
            return _read_file(args.get("path", ""))
        return f"[agent] Unknown tool: {name}"

    # ── main loop ─────────────────────────────────────────────────────────────
    async def run(self, task: str, system: str | None = None) -> dict:
        """
        Run the agent on a single task.
        Returns: { "answer": str, "steps": int, "messages": list }
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": task})

        for step in range(MAX_STEPS):
            msg = _call_model(messages)
            messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                # No tool calls → final answer
                return {"answer": msg.get("content", ""), "steps": step + 1, "messages": messages}

            for tc in tool_calls:
                result = await self._execute(
                    tc["function"]["name"],
                    json.loads(tc["function"]["arguments"]),
                )
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content": [{"type": "text", "text": result, "name": tc["function"]["name"]}],
                })

        # Hit MAX_STEPS
        last = next((m.get("content", "") for m in reversed(messages) if m["role"] == "assistant"), "")
        return {"answer": last, "steps": MAX_STEPS, "messages": messages}


# ── Sync convenience wrapper ──────────────────────────────────────────────────
def run_sync(task: str, system: str | None = None) -> dict:
    """Blocking wrapper — handy for REPL / one-off calls."""
    async def _inner():
        async with Agent() as agent:
            return await agent.run(task, system)
    return asyncio.run(_inner())


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the capital of Japan?"
    r = run_sync(q)
    print(r["answer"])
    print(f"\n[{r['steps']} step(s)]")
