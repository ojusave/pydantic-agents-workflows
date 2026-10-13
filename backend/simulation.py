"""Explicit provider substitutes for the demo and local integration checks.

These fixed answers exercise real application code without contacting a model
provider. The ordinary workflows.app entrypoint never enables these substitutes.
"""

# These standard tools let tests build responses, record worker processes, and
# replace model constructors before the application creates its seven agents.
import asyncio
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

# FunctionModel is Pydantic AI's supported local model substitute. Message parts
# represent plain answer text or the structured fields an agent must return.
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RequestUsage

# All fake embeddings point in the same direction so seeded passages match.
# This checks the database path, not the quality of document search.
VECTOR = [1.0] + [0.0] * 1535
STATE = Path(".local")


def record(kind: str, event: str) -> None:
    """Append one small event so the check can prove separate worker execution."""
    STATE.mkdir(exist_ok=True)
    row = {"kind": kind, "event": event, "pid": os.getpid(), "time": time.time()}
    with (STATE / "synthetic-model-calls.jsonl").open("a") as log:
        log.write(json.dumps(row) + "\n")


async def respond(messages, info, *, failure_probes: bool = False):
    """Return valid example output for each agent, including controlled failures."""
    # An output tool describes required fields. Its first property identifies
    # the pipeline stage without changing any of the real agent factories.
    tool = info.output_tools[0] if info.output_tools else None
    fields = tool.parameters_json_schema["properties"] if tool else {}
    kind = next(iter(fields), "answer")
    record(kind, "start")
    prompt = str(messages)

    # These phrases are test controls. A marker file survives worker restarts,
    # allowing exactly one failure before a Render-managed retry succeeds.
    if failure_probes and kind == "answer" and "retry probe" in prompt.lower():
        try:
            with (STATE / "retry-once").open("x"):
                pass
        except FileExistsError:
            pass
        else:
            record(kind, "injected_failure")
            raise RuntimeError("Synthetic transient failure for retry verification")
    if failure_probes and kind == "answer" and "failure probe" in prompt.lower():
        record(kind, "injected_failure")
        raise RuntimeError("Synthetic persistent failure for error verification")

    # A short delay makes concurrent judge calls measurable across processes.
    await asyncio.sleep(0.25)
    if "queries" in fields:
        output = {
            "queries": ["Render web service deployment", "Render Node.js service"]
        }
    elif "claims" in fields:
        output = {
            "claims": (
                []
                if failure_probes and "empty claims probe" in prompt.lower()
                else [
                    "Render supports Node.js web services.",
                    "Web services can deploy from a Git repository.",
                ]
            )
        }
    elif "supported" in fields:
        output = {"supported": True, "confidence": 0.9, "supporting_doc_indices": [1]}
    elif "accuracy_score" in fields:
        output = {"accuracy_score": 90, "errors": [], "corrections": []}
    elif "overall" in fields:
        output = {key: 90 for key in fields if key != "feedback"}
        output["feedback"] = (
            "SIMULATED: fixed test score, not an actual quality assessment."
        )
    else:
        # The visible label prevents this answer being mistaken for a real model
        # result. Pass the empty-claims marker onward only for that edge-case test.
        answer = (
            ("SIMULATED LOCAL TEST. " if failure_probes else "SIMULATED DEMO. ")
            + "Render supports Node.js web services. "
            "Web services can deploy from a Git repository. [1]"
        )
        if failure_probes and "empty claims probe" in prompt.lower():
            answer += " Empty claims probe."
        record(kind, "done")
        return ModelResponse(parts=[TextPart(answer)], usage=RequestUsage())
    record(kind, "done")
    return ModelResponse(parts=[ToolCallPart(tool.name, output)], usage=RequestUsage())


def simulated_model(model_name, **kwargs):
    """Preserve the configured model name while replacing its network behavior."""
    return FunctionModel(respond, model_name=model_name)


async def simulated_query(self, text, **kwargs):
    """Return one deterministic embedding and zero billed tokens for a question."""
    return SimpleNamespace(embeddings=[VECTOR], usage=SimpleNamespace(input_tokens=0))


async def simulated_documents(self, texts, **kwargs):
    """Return one deterministic embedding per claim, with no provider calls."""
    return SimpleNamespace(
        embeddings=[VECTOR for _ in texts], usage=SimpleNamespace(input_tokens=0)
    )
