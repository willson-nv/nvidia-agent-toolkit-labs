#!/usr/bin/env python3
"""Lab 1 — one scope, one tool call, one model call.

Needs nothing: no API key, no network, no GPU. The "model" below is a local async
function, so this exercises the whole Relay runtime without leaving the machine.

    python lab1_quickstart.py
"""
import asyncio

import nemo_relay


def on_event(event) -> None:
    """Every lifecycle event Relay emits arrives here."""
    print(f"  event={event.kind:<6} name={event.name}")


async def search(args):
    """A stand-in tool. Relay does not care what it does."""
    return {"echo": args["query"]}


async def model(request):
    """A stand-in model. Same point."""
    return {"messages": request.content["messages"], "ok": True}


async def main():
    nemo_relay.subscribers.register("lab1-printer", on_event)

    with nemo_relay.scope.scope("demo-agent", nemo_relay.ScopeType.Agent) as handle:
        nemo_relay.scope.event("initialized", handle=handle, data={"lab": 1})

        tool_result = await nemo_relay.tools.execute(
            "search", {"query": "hello"}, search, handle=handle)

        llm_result = await nemo_relay.llm.execute(
            "demo-provider",
            nemo_relay.LLMRequest({}, {"messages": [{"role": "user", "content": "hi"}]}),
            model,
            handle=handle)

        print("\n  tool returned:", tool_result)
        print("  model returned:", llm_result)

    # Subscriber delivery is asynchronous. Without this flush you can reach the end
    # of the program before the events you are trying to look at have been printed.
    # flush_async, not flush: 0.7.3 refuses to block a running event loop
    await nemo_relay.subscribers.flush_async()
    nemo_relay.subscribers.deregister("lab1-printer")


if __name__ == "__main__":
    asyncio.run(main())
