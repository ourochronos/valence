"""Conftest for integration tests."""

import pytest


def pytest_addoption(parser):
    """Add --live-nodes option to pytest."""
    parser.addoption(
        "--live-nodes",
        action="store_true",
        default=False,
        help="Run tests against live Valence nodes",
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "live_nodes: mark test to run only with --live-nodes")


def pytest_collection_modifyitems(config, items):
    """Skip live node tests unless --live-nodes is passed."""
    if config.getoption("--live-nodes"):
        # --live-nodes given, don't skip tests
        return

    skip_live = pytest.mark.skip(reason="Need --live-nodes option to run")
    for item in items:
        if "integration" in str(item.fspath):
            item.add_marker(skip_live)
