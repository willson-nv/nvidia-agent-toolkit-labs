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
from nemo_relay import ToolExecutionInterceptOutcome

TOOL = "search"


def on_event(event) -> None:
    print(f"  event={event.kind:<6} name={event.name} data={getattr(event, 'data', None)}")


async def search(args):
    """The real tool. Note it always receives the real api_key."""
    assert args.get("api_key"), "the real tool sees the real key"
    return {"hits": [f"result for {args['query']}"]}


# --- 1. sanitise: mask the key in emitted events only -----------------------
# guardrail(tool_name, args) -> the payload to RECORD on the start event.
# Observability only: the real tool still receives the real args.
def redact_api_key(tool_name, args):
    return {**args, "api_key": "***redacted***"}


# --- 2. conditional: refuse to run at all -----------------------------------
# guardrail(tool_name, args) -> None to ALLOW, or a rejection message to BLOCK.
# Note it is not a boolean: returning False would allow the call, because False
# is not None. Returning a message both blocks and explains why.
def require_query(tool_name, args):
    if not str(args.get("query", "")).strip():
        return "query must not be empty"
    return None


# --- 3. execution intercept: wrap the real call -----------------------------
# fn(tool_name, args, next_call) -> ToolExecutionInterceptOutcome
# The outcome type is Rust-native and exported from the nemo_relay TOP level,
# not from nemo_relay.intercepts, despite what the register docstring implies.
async def measure(tool_name, args, next_call):
    t0 = time.perf_counter()
    result = await next_call(args)
    print(f"  [measure] {tool_name} took {(time.perf_counter() - t0) * 1000:.1f} ms")
    return ToolExecutionInterceptOutcome(result)


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

    # flush_async, not flush: 0.7.3 refuses to block a running event loop
    await nemo_relay.subscribers.flush_async()
    nemo_relay.subscribers.deregister("lab2-printer")


if __name__ == "__main__":
    asyncio.run(main())
