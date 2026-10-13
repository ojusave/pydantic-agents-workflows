"""Run the real Render task transport with explicitly simulated model answers.

The local database, task scheduling, retries, validation, and app code are real.
Only model and embedding responses are replaced; no provider requests are sent.
Use this module as the command passed to `render workflows dev`.
"""

# These helpers start local services and protect the dedicated test database.
import asyncio
import sys
from unittest.mock import patch
from urllib.parse import urlsplit

from pydantic_ai import Embedder
from pydantic_ai.models.function import FunctionModel

# Reuse fixed answers in local checks, but enable failure phrases only here.
# Hosted demos never rely on a marker file shared across worker processes.
from backend.simulation import (
    VECTOR,
    respond,
    simulated_query as local_query,
    simulated_documents as local_documents,
)


async def local_respond(messages, info):
    """Enable the retry and empty-output controls for local tests only."""
    return await respond(messages, info, failure_probes=True)


def local_model(model_name, **kwargs):
    """Keep the agent name while returning a controllable local model."""
    return FunctionModel(local_respond, model_name=model_name)


async def seed_database():
    """Seed two artificial passages in the isolated local test database."""
    from backend.config import settings
    from backend.database import vector_store

    # Refuse a remote or ordinary database so a typo cannot alter real history.
    location = urlsplit(settings.database_url)
    if (
        location.hostname not in {"127.0.0.1", "localhost"}
        or location.port != 55437
        or location.path != "/pydantic_demo_test"
    ):
        raise RuntimeError(
            "The synthetic fixture requires its dedicated local database"
        )
    await vector_store.initialize()
    try:
        async with vector_store.pool.acquire() as connection:
            await connection.execute("TRUNCATE documents")
        for source, content in [
            (
                "https://render.com/docs/web-services",
                "Render supports Node.js web services.",
            ),
            (
                "https://render.com/docs/deploy-node-express-app",
                "Web services can deploy from a Git repository.",
            ),
        ]:
            await vector_store.insert_document(
                content,
                source,
                "Synthetic fixture",
                VECTOR,
                metadata={"synthetic": True},
            )
    finally:
        await vector_store.close()


if __name__ == "__main__":
    # Load dummy keys only. The deployed entrypoint never imports this module.
    from dotenv import load_dotenv

    load_dotenv(".env.test.local", override=True)
    with (
        patch("pydantic_ai.models.openai.OpenAIChatModel", side_effect=local_model),
        patch("pydantic_ai.models.anthropic.AnthropicModel", side_effect=local_model),
        patch.object(Embedder, "embed_query", local_query),
        patch.object(Embedder, "embed_documents", local_documents),
    ):
        if "--seed" in sys.argv:
            # Configure local-only logging before database operations emit spans.
            import backend.observability  # noqa: F401

            asyncio.run(seed_database())
        elif "--api" in sys.argv:
            # The web gateway uses the same local settings but real HTTP transport.
            import uvicorn

            uvicorn.run("backend.main:app", host="127.0.0.1", port=8040)
        else:
            # Starting the unchanged registry keeps native child tasks, codecs,
            # result validation and Render's retry handling in the tested path.
            from workflows.app import app

            app.start()
