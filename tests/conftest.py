"""
pytest configuration and shared fixtures.
"""
import pytest
import os

# Ensure a dummy API key is set for tests that instantiate LLMClient
@pytest.fixture(autouse=True)
def set_dummy_env():
    os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
    os.environ.setdefault("TRACE_CONSOLE", "false")
    os.environ.setdefault("TRACE_FILE", "false")
