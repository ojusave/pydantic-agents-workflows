"""Run a hosted rehearsal with real Render tasks and simulated provider output.

Use DEMO_MODE=simulated and DATABASE_URL for the dedicated demo database.
Run with --api for the web gateway, or without arguments for the task registry.
The regular workflows.app entrypoint continues to use real model providers.
"""

# Standard helpers read settings, run database setup, and temporarily replace
# provider calls before the app creates agents. No real provider keys are needed.
import asyncio
import os
import sys
from unittest.mock import patch
from urllib.parse import urlsplit

from pydantic_ai import Embedder

from backend.simulation import (
    VECTOR,
    simulated_documents,
    simulated_model,
    simulated_query,
)


async def seed_demo_database():
    """Add two example passages once, preserving existing questions and history."""
    # Read configuration after the entrypoint has installed dummy credentials.
    from backend.config import settings
    from backend.database import vector_store

    # A dedicated database name prevents this helper from adding artificial
    # documents to an ordinary application's collection by mistake.
    if urlsplit(settings.database_url).path != "/pydantic_native_demo_db":
        raise RuntimeError("The hosted simulation requires pydantic_native_demo_db")
    await vector_store.initialize()
    try:
        # Hold one database lock while checking and inserting. This makes startup
        # safe to repeat, including when old and new service instances overlap.
        async with vector_store.pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute("SELECT pg_advisory_xact_lock(7079439)")
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
                    await connection.execute(
                        """INSERT INTO documents (content, source, title, metadata, embedding)
                        SELECT $1, $2, 'Simulated demo passage', '{"synthetic":true}'::jsonb, $3::vector
                        WHERE NOT EXISTS (
                            SELECT 1 FROM documents WHERE source = $2 AND content = $1
                        )""",
                        content,
                        source,
                        str(VECTOR),
                    )
    finally:
        # The web server opens its own connections on its own event loop later.
        await vector_store.close()


if __name__ == "__main__":
    # Requiring an explicit flag keeps a mistyped startup command from silently
    # turning a real application into a simulation. Dummy keys replace any
    # provider settings in this process; no provider requests are made.
    if os.environ.get("DEMO_MODE") != "simulated":
        raise RuntimeError("Set DEMO_MODE=simulated to use this entrypoint")
    os.environ["OPENAI_API_KEY"] = "simulated-placeholder"
    os.environ["ANTHROPIC_API_KEY"] = "simulated-placeholder"
    os.environ["LOGFIRE_TOKEN"] = ""
    with (
        patch("pydantic_ai.models.openai.OpenAIChatModel", side_effect=simulated_model),
        patch(
            "pydantic_ai.models.anthropic.AnthropicModel", side_effect=simulated_model
        ),
        patch.object(Embedder, "embed_query", simulated_query),
        patch.object(Embedder, "embed_documents", simulated_documents),
    ):
        if "--api" in sys.argv:
            # Initialize local console tracing, seed the remote database, then
            # expose the gateway on the network port assigned by Render.
            import backend.observability  # noqa: F401
            import uvicorn

            asyncio.run(seed_demo_database())
            uvicorn.run(
                "backend.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "10000"))
            )
        else:
            # Every cloud task instance enters here, so child model operations
            # use the same simulation while Render handles scheduling and retries.
            from workflows.app import app

            app.start()
