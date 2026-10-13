"""Render Workflows task definitions for the Ask Render Anything Assistant.

This module is the Workflow service entrypoint. Run it with the SDK CLI:

    render-workflows workflows.app:app

The Q&A root calls ordinary Python pipeline functions. Pydantic AI Harness
creates model tasks for the seven named agents, including each claim verifier.
Ingestion keeps its existing explicitly defined Render tasks.
"""

from __future__ import annotations

import asyncio
import functools
import sys
import time
from pathlib import Path

# Ensure the repo root is importable so `backend.*` and `data.scripts.*`
# resolve regardless of the working directory the workflow runs from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logfire
from render import Retry, TaskContext

# Importing observability configures Logfire at module load (same as the web app).
from backend.observability import pipeline_trace, track_pipeline_metrics
from backend.config import settings
from backend.database import vector_store
from backend.ingestion import embed_documents, replace_source
from backend.models import AnswerResponse, EvaluationResult, PipelineStageResult
from backend.pipeline import (
    check_accuracy,
    collapse_sources,
    embed_question,
    extract_claims,
    generate_answer,
    retrieve_documents,
    verify_claims,
)
from backend.pipeline.evaluation import (
    agreement_level,
    build_evaluation_result,
    evaluate_with_anthropic,
    evaluate_with_openai,
)
from data.sources import SOURCES
# Reuse the registry that the agent factories populated when imported above.
from workflows.runtime import app, pipeline_workflows


def flush_on_exit(fn):
    """Force-flush buffered Logfire spans before the task's instance terminates.

    Each explicit task runs on its own short-lived
    Workflows instance. Without an explicit flush, spans buffered by the OTLP
    exporter can be lost when the instance exits, so `/sessions/{id}/logs` returns
    an empty trace. Flushing in a `finally` (success or failure) ensures each
    instance ships its spans for the shared distributed trace.
    """

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        try:
            return await fn(*args, **kwargs)
        finally:
            logfire.force_flush()

    return wrapper


async def _ensure_ready(db: bool = False) -> None:
    """Initialize per-instance dependencies a task relies on.

    Each task run is a fresh instance, so the pgvector connection pool must be
    initialized here rather than assumed from a long-lived process. Model-price
    data needs no initialization because it ships bundled with the ``genai-prices`` package and
    is looked up in-process (no network call).
    """
    if db:
        await vector_store.initialize()


def _stage_result(
    stage: str,
    *,
    cost_usd: float,
    duration_ms: float = 0.0,
    tokens_used: int | None = None,
    model: str | None = None,
    metadata: dict | None = None,
) -> PipelineStageResult:
    """Build a successful ``PipelineStageResult`` for the observability trail."""
    return PipelineStageResult(
        stage=stage,
        success=True,
        duration_ms=duration_ms,
        cost_usd=cost_usd,
        tokens_used=tokens_used,
        model=model,
        metadata=dict(metadata or {}),
    )


async def _rate_quality(rater, question: str, answer: str, doc_count: int) -> dict:
    """Run a judge and return its validated assessment plus measured token cost.

    The function stays in the root task; only its agent model call is dispatched.
    """
    await _ensure_ready()
    result = await rater(question, answer, doc_count)
    evaluation = build_evaluation_result(result["output"], result["model"])
    return {
        "evaluation": evaluation.model_dump(mode="json"),
        "cost_usd": result["cost_usd"],
        "tokens_used": result["input_tokens"] + result["output_tokens"],
    }


# ---------------------------------------------------------------------------
# Q&A pipeline: orchestrator
# ---------------------------------------------------------------------------

# Activating this shared app sends every attached agent's model requests to
# generated tasks. A root retry repeats the entire answer, so keep it disabled;
# transient model failures still use the per-agent retry policies.
@pipeline_workflows.task(timeout_seconds=600, retry=Retry(max_retries=0, wait_duration_ms=1000))
@flush_on_exit
async def run_qa_pipeline(
    ctx: TaskContext,
    question: str,
    session_id: str | None = None,
    client_id: str | None = None,
    progress_token: str | None = None,
) -> dict:
    """Return one seven-stage answer while native tasks run its model operations.

    Embeddings, database retrieval, prompt preparation and validated outputs live
    in this root task. The seven pre-registered Pydantic agents route their model
    requests through RenderWorkflows. Accuracy and both quality judges can wait
    concurrently, and each claim-verifier request is separately retryable.
    """
    await _ensure_ready(db=True)

    async with pipeline_trace(question):
        # The native integration propagates model-call context to its workers.
        # Retrieval and result assembly remain in this root Python task.
        stages: list[PipelineStageResult] = []
        total_cost = 0.0
        pipeline_start = time.time()

        # Live per-stage feedback. Because the pipeline runs out-of-band in the
        # Workflows service (the gateway only polls for terminal status), we can't
        # stream events as the original in-process pipeline did. Instead each stage
        # appends to a cumulative list persisted under `progress_token`; the gateway
        # reads it on every poll so the UI advances stage-by-stage in real time.
        # No-ops when no token is supplied (e.g. ingest/manual runs).
        progress: list[dict] = []

        async def emit(stage: str, status: str, message: str, pct: float, cost: float) -> None:
            if not progress_token:
                return
            progress.append({
                "stage": stage,
                "status": status,
                "message": message,
                "progress": round(min(pct, 100.0), 1),
                "cost_so_far": round(cost, 4),
            })
            try:
                await vector_store.record_progress(progress_token, progress)
            except Exception as e:  # noqa: BLE001 - progress is best-effort
                logfire.warning(f"Failed to record progress: {e}")

        # --- Retrieval phase (in-process): embedding + RAG ---
        # Question embedding
        await emit("question_embedding", "started", "Embedding your question...", 5, total_cost)
        embed_result = await embed_question(question)
        stages.append(_stage_result(
            "question_embedding",
            cost_usd=embed_result["cost_usd"],
            tokens_used=embed_result["tokens"],
            model=settings.embedding_model,
            metadata={"embedding_dimensions": len(embed_result["embedding"])},
        ))
        total_cost += embed_result["cost_usd"]
        await emit("question_embedding", "completed", "Question embedded", 12.5, total_cost)

        # RAG retrieval (query expansion uses its own native model task)
        await emit("rag_retrieval", "started", "Searching documentation...", 15, total_cost)
        retrieval_result = await retrieve_documents(
            embed_result["embedding"], original_question=question
        )
        stages.append(_stage_result(
            "rag_retrieval",
            cost_usd=retrieval_result["cost_usd"],
            model=settings.query_expansion_model,
            metadata={
                "documents_retrieved": len(retrieval_result["documents"]),
                "queries_expanded": retrieval_result.get("queries_count", 1),
            },
        ))
        total_cost += retrieval_result["cost_usd"]
        documents = retrieval_result["documents"]
        await emit(
            "rag_retrieval", "completed",
            f"Found {len(documents)} relevant documents", 25, total_cost,
        )

        # Single linear pass: generate, ground (claims → verify), then review
        # accuracy + quality. Progress advances across a fixed 28%-95% band.

        # --- Generate phase: the Claude model request as a retried native task ---
        await emit("generation", "started", "Generating answer...", 28, total_cost)
        # agent.run inside this function now dispatches answer__model.request.
        gen_result = await generate_answer(question, documents)
        stages.append(_stage_result(
            "answer_generation",
            cost_usd=gen_result["cost_usd"],
            tokens_used=gen_result["input_tokens"] + gen_result["output_tokens"],
            model=settings.answer_model,
            metadata={"answer_length": len(gen_result["answer"])},
        ))
        total_cost += gen_result["cost_usd"]
        answer_text = gen_result["answer"]
        await emit("generation", "completed", "Answer generated", 37, total_cost)

        # --- Grounding phase: extract claims, then verify them against sources ---
        # Claims extraction (native model task)
        await emit("claims", "started", "Extracting factual claims...", 43, total_cost)
        # Pydantic validates the returned ClaimsOutput after the model task completes.
        claims_result = await extract_claims(answer_text)
        stages.append(_stage_result(
            "claims_extraction",
            cost_usd=claims_result["cost_usd"],
            tokens_used=claims_result["input_tokens"] + claims_result["output_tokens"],
            model=settings.claims_model,
            metadata={"claims_extracted": len(claims_result["claims"])},
        ))
        total_cost += claims_result["cost_usd"]
        await emit(
            "claims", "completed",
            f"Extracted {len(claims_result['claims'])} claims", 49, total_cost,
        )

        # Claims verification (one native model task per claim)
        await emit("verification", "started", "Verifying claims...", 55, total_cost)
        # Database lookups and embeddings stay here. Each claim-verifier agent
        # request becomes its own child task, with independent retries.
        verification_result = await verify_claims(claims_result["claims"])
        verified_claims = verification_result["verified_claims"]
        verification_rate = verification_result["verification_rate"] * 100
        verified_count = len([c for c in verified_claims if c.verified])
        stages.append(_stage_result(
            "claims_verification",
            cost_usd=verification_result["cost_usd"],
            tokens_used=verification_result["tokens_used"],
            model=settings.embedding_model,
            metadata={
                "claims_verified": verified_count,
                "total_claims": len(verified_claims),
                "verification_rate": f"{verification_rate:.0f}%",
            },
        ))
        total_cost += verification_result["cost_usd"]
        await emit(
            "verification", "completed",
            f"{verification_rate:.0f}% claims verified", 61, total_cost,
        )

        # These three reviews do not depend on each other. Their model requests
        # run as concurrent child tasks; validated scores are combined here.
        await emit(
            "accuracy", "started",
            "Checking accuracy & quality in parallel...", 67, total_cost,
        )
        stage_start = time.time()
        # These coroutines wait concurrently. The integration creates separate
        # accuracy, OpenAI-quality, and Claude-quality model tasks automatically.
        accuracy_result, openai_rate, anthropic_rate = await asyncio.gather(
            check_accuracy(answer_text, verified_claims),
            _rate_quality(evaluate_with_openai, question, answer_text, len(documents)),
            _rate_quality(evaluate_with_anthropic, question, answer_text, len(documents)),
        )
        parallel_duration = (time.time() - stage_start) * 1000

        accuracy_score = accuracy_result["accuracy_score"]
        openai_eval = EvaluationResult.model_validate(openai_rate["evaluation"])
        anthropic_eval = EvaluationResult.model_validate(anthropic_rate["evaluation"])
        evaluations = [openai_eval, anthropic_eval]
        average_score = (openai_eval.score + anthropic_eval.score) / 2
        agreement = agreement_level(abs(openai_eval.score - anthropic_eval.score))
        eval_cost = openai_rate["cost_usd"] + anthropic_rate["cost_usd"]
        eval_tokens = openai_rate["tokens_used"] + anthropic_rate["tokens_used"]
        stages.append(_stage_result(
            "technical_accuracy",
            duration_ms=parallel_duration,
            cost_usd=accuracy_result["cost_usd"],
            tokens_used=accuracy_result["input_tokens"] + accuracy_result["output_tokens"],
            model=settings.accuracy_model,
            metadata={"accuracy_score": accuracy_score},
        ))
        total_cost += accuracy_result["cost_usd"]
        await emit(
            "accuracy", "completed",
            f"Accuracy score: {accuracy_score}/100", 79, total_cost,
        )
        stages.append(_stage_result(
            "quality_evaluation",
            duration_ms=parallel_duration,
            cost_usd=eval_cost,
            tokens_used=eval_tokens,
            model=f"{settings.eval_model_openai} + {settings.eval_model_anthropic}",
            metadata={
                "quality_score": f"{average_score:.1f}",
                "openai_score": openai_eval.score,
                "claude_score": anthropic_eval.score,
                "agreement": agreement,
            },
        ))
        total_cost += eval_cost
        await emit(
            "evaluation", "completed",
            f"Quality score: {average_score:.1f}/100", 90, total_cost,
        )
        await emit("finalize", "completed", "Answer ready", 95, total_cost)

        total_duration_ms = (time.time() - pipeline_start) * 1000

        response = AnswerResponse(
            question=question,
            answer=answer_text,
            # Generation above saw every chunk; the user-facing sources list collapses
            # chunks of one page into a single entry (see collapse_sources).
            sources=collapse_sources(documents),
            claims=verified_claims,
            quality_score=average_score,
            accuracy_score=accuracy_score,
            evaluations=evaluations,
            total_cost=total_cost,
            total_duration_ms=total_duration_ms,
            stages=stages,
            session_id=session_id,
        )

        # Persist the session here (the orchestrator already holds all the data
        # and has DB access), so the gateway need not re-receive everything.
        response.session_id = await _persist_session(response, client_id)

        track_pipeline_metrics(
            question=question,
            total_cost=total_cost,
            total_duration_ms=total_duration_ms,
            quality_score=average_score,
            accuracy_score=accuracy_score,
            session_id=response.session_id,
        )

        return response.model_dump(mode="json")


async def _persist_session(response: AnswerResponse, client_id: str | None = None) -> str | None:
    """Save the completed Q&A session to the database."""
    from opentelemetry import trace

    try:
        current_span = trace.get_current_span()
        trace_id = None
        if current_span and current_span.get_span_context().is_valid:
            trace_id = format(current_span.get_span_context().trace_id, "032x")
        saved_id = await vector_store.save_session(
            question=response.question,
            answer=response.answer,
            sources=[doc.model_dump() for doc in response.sources],
            claims=[c.model_dump() for c in response.claims],
            evaluations=[e.model_dump() for e in response.evaluations],
            quality_score=response.quality_score,
            total_cost=response.total_cost,
            total_duration_ms=response.total_duration_ms,
            trace_id=trace_id,
            stages=[s.model_dump() for s in response.stages],
            client_id=client_id,
        )
        logfire.info(f"Saved session to database: {saved_id}", trace_id=trace_id)
        return saved_id
    except Exception as e:  # noqa: BLE001 - persistence is best-effort
        logfire.error(f"Failed to save session: {e}")
        return None


# ---------------------------------------------------------------------------
# Ingestion: source-oriented, data-driven
#
# Every live source follows the same steps: build docs, embed, replace by source.
# so one parameterized task driven by the `SOURCES` registry covers all of them.
# `ingest_source` is the meaningful unit of work (independently retryable per
# source); `ingest_all` fans them out, where
# the heavier sources (pricing's multi-table parse, the tutorials crawl) make
# cross-instance parallelism worth the spin-up.
# ---------------------------------------------------------------------------

@app.task(timeout_seconds=1800)
@flush_on_exit
async def ingest_core(ctx: TaskContext) -> dict:
    """Core documentation ingest (additive sync). Establishes the base schema/rows.

    Loads the pre-embedded corpus (``data/embeddings/render_docs.json``, built
    offline by ``generate_embeddings.py``) into pgvector, so deploy-time ingest
    pays no embedding cost.
    """
    from data.scripts import ingest_docs

    await ingest_docs.main(sync=True)
    return {"task": "ingest_core", "status": "ok"}


@app.task(retry=Retry(max_retries=2, wait_duration_ms=1000))
@flush_on_exit
async def ingest_source(ctx: TaskContext, name: str) -> dict:
    """Ingest one live source end-to-end: build → embed → replace-by-source.

    ``name`` is a key in the ``data.sources.SOURCES`` registry. Each source is an
    independently retryable unit. A transient fetch/embed failure retries just
    this source rather than re-running the whole corpus refresh.
    """
    await _ensure_ready(db=True)
    src = SOURCES[name]
    docs = await embed_documents(await src.build())
    inserted = await replace_source(src.source_url, docs, legacy=src.legacy_sources)
    return {"task": name, "status": "ok", "inserted": inserted}


@app.task(timeout_seconds=3600)
@flush_on_exit
async def ingest_all(ctx: TaskContext) -> dict:
    """Run the full ingestion: core sync, then the live sources fanned out.

    ``ingest_core`` runs first to establish the base schema/rows, then every
    source in the registry is ingested concurrently via ``ingest_source``.
    """
    core = await ctx.run(ingest_core)

    page_results = await asyncio.gather(
        *(ctx.run(ingest_source, name) for name in SOURCES)
    )

    return {"core": core, "pages": list(page_results)}


if __name__ == "__main__":
    app.start()
