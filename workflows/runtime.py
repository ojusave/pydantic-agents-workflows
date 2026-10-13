"""Share one task registry between the pipeline and its Pydantic AI agents."""

# Render owns the list of callable tasks. Retry describes how many additional
# attempts a failed task may make; every cloud task uses the paid standard plan.
from render import Retry, Workflows

# RenderWorkflows is the newly merged Pydantic capability. A capability adds
# behavior to an agent, here sending its model operations to Render workers.
from pydantic_ai_harness import RenderWorkflows

# Importing this module always returns this same app, avoiding a circular import
# between the agent constructors and the pipeline that calls those agents.
app = Workflows(
    default_retry=Retry(max_retries=2, wait_duration_ms=1000, backoff_scaling=2.0),
    default_timeout=300,
    default_plan="standard",
)

# This object activates Render for the root task. Each agent receives its own
# capability below, but all capabilities share this exact app and task context.
pipeline_workflows = RenderWorkflows(app)
