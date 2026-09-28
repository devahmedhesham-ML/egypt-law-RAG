"""Tests never send traces: Langfuse is disabled before any rag module creates its client."""

import os

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"
