#!/usr/bin/env python3
"""
Stripped-down agent harness. ~150 lines. Inspired by mini-swe-agent.

Model   : Minimax API  (MINIMAX_API_KEY)
Search  : minimax-coding-plan-mcp  (same key, spawned once as subprocess)
Tools   : web_search, bash
Loop    : append-only message history; stop when model returns no tool_calls
"""

import asyncio, json, os, re, subprocess, sys
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv
import requests

load_dotenv()
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ── Config (env) ─────────────────────────────────────────────────────────────
MINIMAX_KEY  = os.environ["MINIMAX_API_KEY"]
API_HOST     = os.environ.get("MINIMAX_API_HOST", "https://api.minimax.chat")
MODEL        = os.environ.get("AGENT_MODEL", "MiniMax-2.7")
MAX_STEPS    = int(os.environ.get("AGENT_MAX_STEPS", "100"))

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
            "name": "bash",
            "description": (
                "Run a shell command and return stdout+stderr (max 20000 chars). "
                "Use for: fetching URLs (curl), parsing HTML/JSON (python3 -c), "
                "reading/writing files, data processing. Timeout: 120s."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"}
                },
                "required": ["command"],
            },
        },
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _strip_thinking(text: str) -> str:
    """Remove <think>…</think> blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _log(step: int, tag: str, body: str, verbose: bool) -> None:
    if not verbose:
        return
    GRAY, CYAN, GREEN, YELLOW, RESET = "\033[90m", "\033[36m", "\033[32m", "\033[33m", "\033[0m"
    colors = {"think": GRAY, "tool_call": CYAN, "tool_result": GREEN, "answer": YELLOW}
    c = colors.get(tag, RESET)
    prefix = f"{GRAY}[step {step}]{RESET} {c}{tag}{RESET}"
    # Truncate long bodies for readability
    lines = body.split("\n")
    if len(lines) > 30:
        body = "\n".join(lines[:25]) + f"\n{GRAY}... ({len(lines) - 25} more lines){RESET}"
    print(f"{prefix}\n{body}\n", file=sys.stderr)


# ── Tool implementations ──────────────────────────────────────────────────────
def _run_bash(command: str, timeout: int = 120) -> str:
    try:
        r = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True, timeout=timeout,
        )
        out = (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        out = f"[bash] Command timed out after {timeout}s"
    return out[:20000] or "[bash] (no output)"


# ── Minimax API ───────────────────────────────────────────────────────────────
def _call_model(messages: list) -> dict:
    resp = requests.post(
        f"{API_HOST}/v1/chat/completions",
        headers={"Authorization": f"Bearer {MINIMAX_KEY}",
                 "Content-Type": "application/json"},
        json={"model": MODEL, "messages": messages,
              "tools": TOOL_DEFS, "tool_choice": "auto"},
        timeout=300,
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
        if name == "bash":
            return _run_bash(args.get("command", "echo 'no command'"))
        return f"[agent] Unknown tool: {name}"

    # ── main loop ─────────────────────────────────────────────────────────────
    async def run(self, task: str, system: str | None = None, verbose: bool = False) -> dict:
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

            # Log thinking (if present) and text content
            content = msg.get("content", "") or ""
            thinking = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
            if thinking:
                _log(step + 1, "think", thinking.group(1).strip(), verbose)
            clean = _strip_thinking(content)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                _log(step + 1, "answer", clean, verbose)
                return {"answer": clean, "steps": step + 1, "messages": messages}

            for tc in tool_calls:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])
                _log(step + 1, "tool_call", f"{name}({json.dumps(args, ensure_ascii=False)})", verbose)

                result = await self._execute(name, args)
                _log(step + 1, "tool_result", result, verbose)

                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content": [{"type": "text", "text": result, "name": name}],
                })

        # Hit MAX_STEPS
        last = next((m.get("content", "") for m in reversed(messages) if m["role"] == "assistant"), "")
        return {"answer": _strip_thinking(last), "steps": MAX_STEPS, "messages": messages}


# ── Sync convenience wrapper ──────────────────────────────────────────────────
def run_sync(task: str, system: str | None = None, verbose: bool = False) -> dict:
    """Blocking wrapper — handy for REPL / one-off calls."""
    async def _inner():
        async with Agent() as agent:
            return await agent.run(task, system, verbose=verbose)
    return asyncio.run(_inner())


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the capital of Japan?"
    r = run_sync(q, verbose=True)
    print(r["answer"])
    print(f"\n[{r['steps']} step(s)]")
