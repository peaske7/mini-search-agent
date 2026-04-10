#!/usr/bin/env python3
import asyncio, json, os, re, subprocess, sys, threading, time
from contextlib import AsyncExitStack, contextmanager
from dotenv import load_dotenv
import requests
load_dotenv()
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MINIMAX_KEY, API_HOST = os.environ["MINIMAX_API_KEY"], os.environ.get("MINIMAX_API_HOST", "https://api.minimax.chat")
MODEL, MAX_STEPS = os.environ.get("AGENT_MODEL", "MiniMax-2.7"), int(os.environ.get("AGENT_MAX_STEPS", "100"))
DIM, CYAN, GREEN, YELLOW, BOLD, RESET = "\033[2m", "\033[36m", "\033[32m", "\033[33m", "\033[1m", "\033[0m"
COLORS = {"think": DIM, "web_search": CYAN, "bash": GREEN, "result": DIM, "answer": YELLOW}

def _tool(name, desc, param, pdesc):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "required": [param],
                           "properties": {param: {"type": "string", "description": pdesc}}}}}
TOOL_DEFS = [
    _tool("web_search", "Search the live web. Returns titles, URLs, and snippets.", "query", "Search query"),
    _tool("bash", "Run a shell command (stdout+stderr, max 20k chars, 120s timeout). "
          "Use for curl, python3 -c, file I/O.", "command", "Shell command"),
]

def _strip_thinking(text): return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

@contextmanager
def _spinner(label, verbose):
    if not verbose or not sys.stderr.isatty(): yield; return
    stop, t0 = threading.Event(), time.time()
    def _spin():
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        while not stop.wait(0.08):
            e = time.time() - t0
            print(f"\r{DIM}{frames[int(e/0.08) % 10]} {label} ({e:.0f}s){RESET}  ", end="", file=sys.stderr, flush=True)
        print("\r\033[2K", end="", file=sys.stderr, flush=True)
    th = threading.Thread(target=_spin, daemon=True); th.start()
    try: yield
    finally: stop.set(); th.join()

def _log(step, tag, body, verbose):
    if not verbose: return
    lines = body.split("\n")
    if len(lines) > 25: lines = lines[:20] + [f"{DIM}    ... +{len(lines)-20} lines{RESET}"]
    print(f"{DIM}[{step}]{RESET} {COLORS.get(tag, DIM)}{BOLD}{tag}{RESET}\n{DIM}" +
          "\n".join(f"    {l}" for l in lines) + f"{RESET}\n", file=sys.stderr)

def _run_bash(cmd, timeout=120):
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
        out = (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired: out = f"[bash] timed out after {timeout}s"
    return out[:20000] or "[bash] (no output)"

def _call_model(messages):
    r = requests.post(f"{API_HOST}/v1/chat/completions", headers={"Authorization": f"Bearer {MINIMAX_KEY}"},
                      json={"model": MODEL, "messages": messages, "tools": TOOL_DEFS, "tool_choice": "auto"}, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]

class Agent:
    def __init__(self): self._mcp = self._stack = None

    async def __aenter__(self):
        self._stack = AsyncExitStack()
        params = StdioServerParameters(command="uvx", args=["minimax-coding-plan-mcp"],
            env={**os.environ, "MINIMAX_API_KEY": MINIMAX_KEY, "MINIMAX_API_HOST": API_HOST})
        r, w = await self._stack.enter_async_context(stdio_client(params))
        self._mcp = await self._stack.enter_async_context(ClientSession(r, w))
        await self._mcp.initialize()
        return self

    async def __aexit__(self, *_):
        if self._stack: await self._stack.aclose()

    async def _execute(self, name, args):
        if name == "web_search":
            res = await self._mcp.call_tool("web_search", {"query": args.get("query", "")})
            return "\n".join(c.text for c in res.content if hasattr(c, "text")) or "No results."
        if name == "bash": return _run_bash(args.get("command", "echo 'no command'"))
        return f"[agent] Unknown tool: {name}"

    async def run(self, task, system=None, verbose=False):
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": task})
        t0 = time.time()
        for step in range(MAX_STEPS):
            with _spinner(f"step {step+1}", verbose): msg = _call_model(messages)
            messages.append(msg)
            content = msg.get("content", "") or ""
            if (m := re.search(r"<think>(.*?)</think>", content, re.DOTALL)):
                _log(step+1, "think", m.group(1).strip(), verbose)
            if not (tool_calls := msg.get("tool_calls") or []):
                answer = _strip_thinking(content)
                _log(step+1, "answer", answer, verbose)
                if verbose: print(f"{DIM}{'─'*40}\n  {step+1} steps · {time.time()-t0:.1f}s{RESET}\n", file=sys.stderr)
                return {"answer": answer, "steps": step+1, "messages": messages}
            for tc in tool_calls:
                name, args = tc["function"]["name"], json.loads(tc["function"]["arguments"])
                _log(step+1, name, args.get("query", args.get("command", "")), verbose)
                with _spinner(name, verbose): result = await self._execute(name, args)
                _log(step+1, "result", result, verbose)
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": [{"type": "text", "text": result, "name": name}]})
        last = next((m.get("content", "") for m in reversed(messages) if m["role"] == "assistant"), "")
        if verbose: print(f"{DIM}{'─'*40}\n  {MAX_STEPS} steps (max) · {time.time()-t0:.1f}s{RESET}\n", file=sys.stderr)
        return {"answer": _strip_thinking(last), "steps": MAX_STEPS, "messages": messages}

def run_sync(task, system=None, verbose=False):
    async def _go():
        async with Agent() as a: return await a.run(task, system, verbose=verbose)
    return asyncio.run(_go())

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is the capital of Japan?"
    r = run_sync(q, verbose=True)
    print(r["answer"])
    print(f"\n[{r['steps']} step(s)]")
