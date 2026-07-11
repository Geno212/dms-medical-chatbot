import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tests must not leave durable conversation checkpoints behind.
os.environ.setdefault("CHECKPOINT_DB", ":memory:")

from app.config import get_config
from app.db import get_repository


class FakeLLM:
    """Scriptable stand-in for the chat model.

    Recognizes which prompt the engine is sending (router / slot extraction /
    free text) and returns pre-programmed values, so tests exercise the full
    graph, retrieval, and entity-verification logic deterministically.
    """

    def __init__(self, router=None, slots=None, text="This is a grounded medical answer. Would you like to book?"):
        self.router = list(router) if isinstance(router, list) else [router] if router else []
        self.slots = list(slots) if isinstance(slots, list) else [slots] if slots else []
        self.text = text

    def chat(self, system, messages, json_mode=False):
        if "intent router" in system:
            value = self.router.pop(0) if self.router else {"intent": "medical", "action": "none"}
            return json.dumps(value)
        if "extract booking/lookup parameters" in system:
            value = self.slots.pop(0) if self.slots else {"doctor_name": None, "specialty": None, "branch": None}
            return json.dumps(value)
        return self.text


def _scrub_test_appointments(repository) -> None:
    """Remove only the rows the tests created (thread_id starts with 'test-').
    Real demo bookings live under other thread ids and are never touched, so
    the database is left exactly as the tests found it — important because the
    suite runs against the live Supabase backend."""
    for apt in list(repository.list_appointments_like("test-")):
        repository.delete_appointment(apt["id"])


@pytest.fixture(scope="session")
def repo():
    """The configured backend — Supabase/Postgres when APP_DATABASE_URL is set,
    local SQLite otherwise. Tests exercise the same repository the app uses.
    Assumes the backend is already seeded (run scripts/seed_db.py first)."""
    config = get_config()
    repository = get_repository(config)
    _scrub_test_appointments(repository)  # start clean even if a prior run crashed
    yield repository
    _scrub_test_appointments(repository)
    repository.close()
