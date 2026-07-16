import os

import pytest

# The test suite must never depend on a Hugging Face download or an API key.
os.environ.setdefault("BIOENHANCE_RETRIEVER", "tfidf")


def pytest_configure(config):
    config.addinivalue_line("markers", "network: test requires internet access")


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that hit external APIs (PubChem, PubMed)",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
