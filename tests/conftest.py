import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tests must not leave durable conversation checkpoints behind.
os.environ.setdefault("CHECKPOINT_DB", ":memory:")

from app.config import get_config
from app.db import Repository
from scripts.seed_db import seed


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


@pytest.fixture(scope="session")
def repo():
    config = get_config()
    if not Path(config.db_path).exists():
        seed(skip_embeddings=True)
    repository = Repository(config.db_path)
    yield repository
    # Booking tests write real rows; scrub anything test threads created.
    repository.conn().execute("DELETE FROM appointments WHERE thread_id LIKE 'test-%'")
    repository.conn().commit()
    repository.close()
