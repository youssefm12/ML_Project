"""
tests/conftest.py
=============================================================================
Pytest Configuration and Shared Fixtures.
Provides the Flask test client and ensures the AI registry is hydrated during tests.
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.app import create_app
from app.models import registry

@pytest.fixture(scope="session")
def app():
    """Create and configure a new Flask instance for testing."""
    _app = create_app()
    _app.config.update({
        "TESTING": True,
    })
    
    # Ensure artifacts are loaded into memory for integration testing
    registry.load_artifacts()
    
    yield _app

@pytest.fixture(scope="session")
def client(app):
    """A test client for the app."""
    return app.test_client()

@pytest.fixture(scope="session")
def runner(app):
    """A test runner for the app's Click commands."""
    return app.test_cli_runner()
