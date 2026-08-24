"""Unit tests for the support-triage verifier.

These call no model and need no GPU, no API key and no network. This is the inner
loop: change verify(), run pytest, know immediately whether the reward is right.

    gym env test --resources-server support_triage
"""
import json
from types import SimpleNamespace

import pytest

from app import SupportTriageResourcesServer, SupportTriageVerifyRequest


def _response(text: str):
    """Minimal stand-in for a Responses-API result carrying one assistant message."""
    return SimpleNamespace(output=[SimpleNamespace(
        type="message",
        content=[SimpleNamespace(type="output_text", text=text)])])


def _request(text, severity="P0", team="infra"):
    return SupportTriageVerifyRequest(
        responses_create_params={"input": []},
        response=_response(text),
        verifier_metadata={"severity": severity, "team": team},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("payload,expected", [
    ('{"severity":"P0","team":"infra"}', 1.0),    # both right
    ('{"severity":"P0","team":"billing"}', 0.5),  # severity only
    ('{"severity":"P2","team":"infra"}', 0.5),    # team only
    ('{"severity":"P2","team":"billing"}', 0.0),  # both wrong
    ('{"severity":"URGENT","team":"infra"}', 0.5),  # invalid enum, team still right
    ('I think this is an infrastructure problem.', 0.0),  # not JSON
    ('', 0.0),                                    # nothing at all
])
async def test_reward(payload, expected):
    server = SupportTriageResourcesServer.__new__(SupportTriageResourcesServer)
    out = await server.verify(_request(payload))
    assert out.reward == expected


@pytest.mark.asyncio
async def test_ground_truth_survives_parsing():
    """The regression test for the silent-drop trap.

    If verifier_metadata is ever removed from the request subclass this fails,
    which is the whole point -- otherwise the bug is invisible.
    """
    req = _request('{"severity":"P1","team":"auth"}', severity="P1", team="auth")
    assert req.verifier_metadata == {"severity": "P1", "team": "auth"}
    server = SupportTriageResourcesServer.__new__(SupportTriageResourcesServer)
    assert (await server.verify(req)).reward == 1.0
