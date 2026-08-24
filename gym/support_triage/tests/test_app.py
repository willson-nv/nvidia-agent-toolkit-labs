#!/usr/bin/env python3
"""Tests for the support_triage verifier.

    gym env test --resources-server support_triage

Seven parametrised reward cases plus one regression test for the dropped-ground-
truth trap that step 7 of the lab creates on purpose.
"""
from unittest.mock import MagicMock

import pytest
from app import (
    SupportTriageResourcesServer,
    SupportTriageResourcesServerConfig,
    SupportTriageVerifyRequest,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def _response(text: str) -> NeMoGymResponse:
    """A minimal but VALID OpenAI Responses object.

    `NeMoGymResponse` subclasses OpenAI's `Response`, so eight fields are
    required -- id, created_at, model, object, output, parallel_tool_calls,
    tool_choice, tools -- even though a verifier only ever reads `output`.

    Do not try to shortcut this with a SimpleNamespace or a bare dict. Pydantic
    rejects it with "Input should be a valid dictionary or instance of
    NeMoGymResponse", and every test fails in the fixture before your verify()
    is ever called. Copy the shape from a shipped environment's tests instead of
    reverse-engineering it.
    """
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _server() -> SupportTriageResourcesServer:
    return SupportTriageResourcesServer(
        config=SupportTriageResourcesServerConfig(
            host="0.0.0.0", port=8080, entrypoint="", name=""
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def _request(text: str, severity: str = "P0", team: str = "infra") -> SupportTriageVerifyRequest:
    return SupportTriageVerifyRequest(
        responses_create_params={
            "input": [{"role": "user", "content": "a ticket"}],
            "parallel_tool_calls": False,
            "temperature": 0,
        },
        response=_response(text),
        verifier_metadata={"severity": severity, "team": team},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,expected",
    [
        ('{"severity":"P0","team":"infra"}', 1.0),             # both right
        ('{"severity":"P0","team":"billing"}', 0.5),           # severity only
        ('{"severity":"P2","team":"infra"}', 0.5),             # team only
        ('{"severity":"P2","team":"billing"}', 0.0),           # both wrong
        ('{"severity":"URGENT","team":"infra"}', 0.5),         # invalid enum, team still right
        ("I think this is an infrastructure problem.", 0.0),   # not JSON at all
        ("", 0.0),                                            # nothing at all
    ],
)
async def test_reward(payload: str, expected: float) -> None:
    out = await _server().verify(_request(payload))
    assert out.reward == expected


@pytest.mark.asyncio
async def test_ground_truth_survives_parsing() -> None:
    """The regression test for the dropped-ground-truth trap.

    Pydantic drops any field on a dataset row that the request model does not
    declare. Remove `verifier_metadata` from SupportTriageVerifyRequest and
    verify() raises AttributeError -- a 500 from the server, and a dead eval run.

    The catch is that the sad-path tests above keep passing when that happens,
    because verify() returns early on unparseable output and never reads ground
    truth. This test is the one that reaches the comparison, so it is the one
    that fails.
    """
    req = _request('{"severity":"P1","team":"auth"}', severity="P1", team="auth")
    assert req.verifier_metadata == {"severity": "P1", "team": "auth"}

    out = await _server().verify(req)
    assert out.reward == 1.0
