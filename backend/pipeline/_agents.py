"""Construct named agents whose model requests use native Render tasks."""

# Type hints describe the allowed structured output, for example ClaimsOutput.
from typing import Any

# Pydantic AI drives each agent; provider objects read credentials in the worker.
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider

# Options sets limits for each generated model task. The capability performs
# transport and result validation using the upstream Pydantic integration.
from pydantic_ai_harness import RenderWorkflows
from render import Options, Retry

# Settings holds local environment values. The shared app collects every task
# before Render starts workers; agent names keep generated task names distinct.
from backend.config import settings
from workflows.runtime import app


def _capability(name: str, timeout_seconds: int, max_retries: int) -> RenderWorkflows:
    """Give one agent its own registered model tasks and bounded retry policy."""
    return RenderWorkflows(
        app,
        name=name,
        model_options=Options(
            timeout_seconds=timeout_seconds,
            retry=Retry(
                max_retries=max_retries, wait_duration_ms=1000, backoff_scaling=2.0
            ),
            plan="standard",
        ),
    )


def anthropic_agent(
    model: str,
    instructions: str,
    output_type: Any = str,
    *,
    name: str,
    timeout_seconds: int = 90,
    max_retries: int = 2,
) -> Agent:
    """Create a Claude agent; its prompt enters here and validated output returns to the caller."""
    # Each worker rebuilds its own provider client from its environment. No API
    # key is placed in the arguments sent from a parent task to a child task.
    return Agent(
        AnthropicModel(
            model, provider=AnthropicProvider(api_key=settings.anthropic_api_key)
        ),
        name=name,
        output_type=output_type,
        instructions=instructions,
        capabilities=[_capability(name, timeout_seconds, max_retries)],
    )


def openai_agent(
    model: str,
    instructions: str,
    output_type: Any = str,
    *,
    name: str,
    timeout_seconds: int = 60,
    max_retries: int = 2,
) -> Agent:
    """Create an OpenAI agent with the same native task boundary as the Claude agents."""
    # Construction happens when the module loads, so all generated task names
    # exist before the worker accepts a question. Outside an activated task,
    # ordinary agent.run calls still work in the current Python process.
    return Agent(
        OpenAIChatModel(
            model, provider=OpenAIProvider(api_key=settings.openai_api_key)
        ),
        name=name,
        output_type=output_type,
        instructions=instructions,
        capabilities=[_capability(name, timeout_seconds, max_retries)],
    )
