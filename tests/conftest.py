# This file runs automatically before any test. It does three simple things:
#   1. lets tests import the project code (clients, dataio, detector, ...)
#   2. puts FAKE api keys in place, so the real keys in .env are never loaded
#   3. blocks the internet, so no test can ever call a real (paid) model API

import os
import socket
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. make the project importable. detector/ and paper_analysis/ scripts import
#    their neighbours by bare name (e.g. "from number_teeth import ..."),
#    so those folders go on the path too.
for folder in [REPO, os.path.join(REPO, "detector"), os.path.join(REPO, "paper_analysis")]:
    if folder not in sys.path:
        sys.path.insert(0, folder)

# 2. fake keys. load_dotenv() never overwrites a variable that is already set,
#    so with these in place the real keys in .env stay unused.
for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]:
    os.environ[key] = "fake-key-for-tests"


# 3. no internet during tests
class NoInternet(Exception):
    pass


def refuse_to_connect(*args, **kwargs):
    raise NoInternet("tests are not allowed to use the internet")


@pytest.fixture(autouse=True)
def block_internet(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", refuse_to_connect)
    monkeypatch.setattr(socket, "create_connection", refuse_to_connect)
