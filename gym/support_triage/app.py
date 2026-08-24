#!/usr/bin/env python3
"""Lab 5 — a NeMo Gym environment for support-ticket triage.

A ticket arrives as free text. The model must answer with JSON:

    {"severity": "P0"|"P1"|"P2", "team": "billing"|"auth"|"infra"}

Single step, no tools, verify only. The whole environment is the verify() method
at the bottom; everything else is boilerplate the scaffold generated.

    gym env start --resources-server support_triage --model-type inference_provider
"""
import json
from typing import Any, Optional

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)

SEVERITIES = {"P0", "P1", "P2"}
TEAMS = {"billing", "auth", "infra"}


class SupportTriageResourcesServerConfig(BaseResourcesServerConfig):
    pass


class SupportTriageVerifyRequest(BaseVerifyRequest):
    """THIS SUBCLASS IS LOAD-BEARING.

    Fields on a dataset row that are not declared on the request model are
    dropped by Pydantic -- BaseVerifyRequest.model_config is {}, so plain
    extra="ignore". Without `verifier_metadata` declared here, the ground truth
    never reaches verify() and `body.verifier_metadata` raises AttributeError,
    which the server returns as a 500 and which kills a `gym eval run` on row 0.

    Measured, not assumed: `gym env test` gives 6 failed, 2 passed. The two that
    still pass are the prose case and the empty-string case -- the ones asserting
    reward == 0.0 -- because verify() returns early before reading ground truth.
    A verifier with a completely destroyed ground-truth path still passes every
    test that expects a zero.

    Lab 5 deletes this line on purpose, shows the damage, then puts it back.
    """
    verifier_metadata: dict[str, Any]


class SupportTriageVerifyResponse(BaseVerifyResponse):
    parsed: Optional[dict[str, Any]] = None


class SupportTriageResourcesServer(SimpleResourcesServer):
    config: SupportTriageResourcesServerConfig

    @staticmethod
    def _assistant_text(response) -> str:
        """Flatten the model's output items into one string.

        There is no shared helper for this in nemo_gym -- most shipped
        environments hand-roll the same dozen lines.
        """
        parts = []
        for item in response.output:
            if getattr(item, "type", None) != "message":
                continue
            for chunk in item.content:
                if getattr(chunk, "type", None) == "output_text":
                    parts.append(chunk.text)
        return "".join(parts)

    async def verify(self, body: SupportTriageVerifyRequest) -> SupportTriageVerifyResponse:
        text = self._assistant_text(body.response)

        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            # Not JSON at all. No partial credit for prose.
            return SupportTriageVerifyResponse(**body.model_dump(), reward=0.0, parsed=None)

        truth = body.verifier_metadata
        hits = 0
        if parsed.get("severity") in SEVERITIES and parsed.get("severity") == truth.get("severity"):
            hits += 1
        if parsed.get("team") in TEAMS and parsed.get("team") == truth.get("team"):
            hits += 1

        # Partial credit, on purpose. Pass/fail gives a training loop almost
        # nothing to work with; "one of two fields right" is a gradient.
        return SupportTriageVerifyResponse(
            **body.model_dump(), reward=hits / 2.0, parsed=parsed)


if __name__ == "__main__":
    SupportTriageResourcesServer.run_webserver()
