#!/usr/bin/env python3
"""Lab 2 — three middlewares on one tool: redact, reject, measure.

Still no API key, no network, no GPU.

The three registrations at the bottom are the whole lesson:

  * a SANITISE guardrail changes what is *emitted*, never what executes
  * a CONDITIONAL guardrail can stop the call before the tool function is entered
  * an EXECUTION intercept wraps the real call

    python lab2_middleware.py

NOTE: callback signatures are the part of this file most likely to drift between
Relay releases. If a registration raises a TypeError, check the middleware guide
for the installed version before changing anything else.
"""
import asyncio
import time

import nemo_relay

TOOL = "search"


def on_event(event) -> None:
    print(f"  event={event.kind:<6} name={event.name} data={getattr(event, 'data', None)}")


async def search(args):
    """The real tool. Note it always receives the real api_key."""
    assert args.get("api_key"), "the real tool sees the real key"
    return {"hits": [f"result for {args['query']}"]}


# --- 1. sanitise: mask the key in emitted events only -----------------------
def redact_api_key(request):
    scrubbed = dict(request.content)
    if "api_key" in scrubbed:
        scrubbed["api_key"] = "***redacted***"
    return scrubbed


# --- 2. conditional: refuse to run at all -----------------------------------
def require_query(request):
    """Return False to block. The tool function is never entered."""
    return bool(str(request.content.get("query", "")).strip())


# --- 3. execution intercept: wrap the real call -----------------------------
async def measure(request, next_):
    t0 = time.perf_counter()
    result = await next_(request)
    print(f"  [measure] {TOOL} took {(time.perf_counter() - t0) * 1000:.1f} ms")
    return result


async def call(handle, query):
    args = {"query": query, "api_key": "sk-fake-not-a-real-key"}
    try:
        out = await nemo_relay.tools.execute(TOOL, args, search, handle=handle)
        print(f"  -> allowed: {out}")
    except Exception as exc:                      # blocked calls fail closed
        print(f"  -> blocked: {type(exc).__name__}: {exc}")


async def main():
    nemo_relay.subscribers.register("lab2-printer", on_event)

    nemo_relay.guardrails.register_tool_sanitize_request(
        f"{TOOL}.redact_api_key", 10, redact_api_key)
    nemo_relay.guardrails.register_tool_conditional_execution(
        f"{TOOL}.require_query", 20, require_query)
    nemo_relay.intercepts.register_tool_execution(
        f"{TOOL}.measure", 30, measure)

    with nemo_relay.scope.scope("demo-agent", nemo_relay.ScopeType.Agent) as handle:
        print("\n--- a good call ------------------------------------------")
        await call(handle, "hello")
        print("\n--- an empty query ---------------------------------------")
        await call(handle, "   ")

    nemo_relay.subscribers.flush()
    nemo_relay.subscribers.deregister("lab2-printer")


if __name__ == "__main__":
    asyncio.run(main())
