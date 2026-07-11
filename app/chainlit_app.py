"""Chainlit web UI.

Run:  chainlit run app/chainlit_app.py -w

* Text chat in Arabic or English; login with the demo credentials
  (CHAT_USER / CHAT_PASSWORD env vars, default demo / demo).
* Voice: press the microphone button (streamed PCM is buffered and
  transcribed locally with faster-whisper), or attach an audio file to a
  message. The transcription is echoed as {"transcribed_text": ...} and then
  processed as a regular chat message — exactly as the task specifies.
* Medical answers render as chat text. Action responses render as a
  human-readable confirmation (deterministically derived from the verified
  payload — no LLM) with the structured JSON payload shown underneath.
* Conversations persist: chat history lives in data/chainlit.db (Chainlit
  data layer) and the graph state (context memory, clinical slot-filling)
  in data/conversations.db (LangGraph checkpoints) — reopening a thread
  from the sidebar resumes it with full memory, even across restarts.
"""

from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import uuid
import wave
from pathlib import Path

# Ensure the project root (medical-chatbot/) is on sys.path so that
# `app.*` imports work whether this file is loaded by Chainlit directly
# (as a file path) or as part of the package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.types import ThreadDict

from app.config import get_config
from app.graph.build import build_graph, chat_turn
from app.graph.nodes import detect_language
from app.presenter import humanize

graph = build_graph()

# --------------------------------------------------------------- persistence

CHAINLIT_DB = _PROJECT_ROOT / "data" / "chainlit.db"

# Chainlit's SQLAlchemy data layer expects this schema to exist (it does not
# create it). Column set mirrors chainlit 2.11 StepDict/ThreadDict/ElementDict.
_HISTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL DEFAULT '{}',
    "createdAt" TEXT
);
CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    "tags" TEXT,
    "metadata" TEXT,
    "favorite" INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT,
    "type" TEXT,
    "threadId" TEXT,
    "parentId" TEXT,
    "command" TEXT,
    "modes" TEXT,
    "streaming" INTEGER,
    "waitForAnswer" INTEGER,
    "isError" INTEGER,
    "metadata" TEXT,
    "tags" TEXT,
    "input" TEXT,
    "output" TEXT,
    "createdAt" TEXT,
    "start" TEXT,
    "end" TEXT,
    "generation" TEXT,
    "showInput" TEXT,
    "defaultOpen" INTEGER,
    "autoCollapse" INTEGER,
    "language" TEXT,
    "icon" TEXT,
    "feedback" TEXT
);
CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT,
    "type" TEXT,
    "chainlitKey" TEXT,
    "path" TEXT,
    "url" TEXT,
    "objectKey" TEXT,
    "name" TEXT,
    "display" TEXT,
    "size" TEXT,
    "language" TEXT,
    "page" INTEGER,
    "props" TEXT,
    "autoPlay" INTEGER,
    "playerConfig" TEXT,
    "forId" TEXT,
    "mime" TEXT
);
CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT,
    "threadId" TEXT,
    "value" INTEGER,
    "comment" TEXT
);
CREATE INDEX IF NOT EXISTS idx_steps_thread ON steps("threadId");
CREATE INDEX IF NOT EXISTS idx_elements_thread ON elements("threadId");
"""


# Postgres flavor of the same schema: TEXT[] where chainlit binds Python
# lists, BOOLEAN/INTEGER where it binds bools/ints, TEXT everywhere else
# (JSON fields arrive pre-serialized as strings).
_HISTORY_SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS users (
    "id" TEXT PRIMARY KEY,
    "identifier" TEXT NOT NULL UNIQUE,
    "metadata" TEXT NOT NULL DEFAULT '{}',
    "createdAt" TEXT
);
CREATE TABLE IF NOT EXISTS threads (
    "id" TEXT PRIMARY KEY,
    "createdAt" TEXT,
    "name" TEXT,
    "userId" TEXT,
    "userIdentifier" TEXT,
    "tags" TEXT[],
    "metadata" TEXT,
    "favorite" BOOLEAN DEFAULT FALSE
);
CREATE TABLE IF NOT EXISTS steps (
    "id" TEXT PRIMARY KEY,
    "name" TEXT, "type" TEXT, "threadId" TEXT, "parentId" TEXT,
    "command" TEXT, "modes" TEXT,
    "streaming" BOOLEAN, "waitForAnswer" BOOLEAN, "isError" BOOLEAN,
    "metadata" TEXT, "tags" TEXT[], "input" TEXT, "output" TEXT,
    "createdAt" TEXT, "start" TEXT, "end" TEXT, "generation" TEXT,
    "showInput" TEXT, "defaultOpen" BOOLEAN, "autoCollapse" BOOLEAN,
    "language" TEXT, "icon" TEXT, "feedback" TEXT
);
CREATE TABLE IF NOT EXISTS elements (
    "id" TEXT PRIMARY KEY,
    "threadId" TEXT, "type" TEXT, "chainlitKey" TEXT, "path" TEXT,
    "url" TEXT, "objectKey" TEXT, "name" TEXT, "display" TEXT,
    "size" TEXT, "language" TEXT, "page" INTEGER, "props" TEXT,
    "autoPlay" BOOLEAN, "playerConfig" TEXT, "forId" TEXT, "mime" TEXT
);
CREATE TABLE IF NOT EXISTS feedbacks (
    "id" TEXT PRIMARY KEY,
    "forId" TEXT, "threadId" TEXT, "value" INTEGER, "comment" TEXT
);
CREATE INDEX IF NOT EXISTS idx_steps_thread ON steps("threadId");
CREATE INDEX IF NOT EXISTS idx_elements_thread ON elements("threadId");
"""

_config = get_config()
_USE_PG_HISTORY = _config.db_backend == "postgres" and bool(_config.database_url)


def _ensure_history_schema() -> None:
    if _USE_PG_HISTORY:
        import psycopg2

        conn = psycopg2.connect(_config.database_url)
        try:
            with conn.cursor() as cur:
                cur.execute(_HISTORY_SCHEMA_PG)
            conn.commit()
        finally:
            conn.close()
        return
    CHAINLIT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHAINLIT_DB))
    try:
        conn.executescript(_HISTORY_SCHEMA)
        conn.commit()
    finally:
        conn.close()


_ensure_history_schema()


def _history_conninfo() -> str:
    """Chat history follows the data backend: Supabase/Postgres (via the
    async driver) when DB_BACKEND=postgres, local SQLite otherwise."""
    if _USE_PG_HISTORY:
        url = _config.database_url
        if url.startswith("postgresql://"):
            url = "postgresql+asyncpg://" + url[len("postgresql://"):]
        return url
    return f"sqlite+aiosqlite:///{CHAINLIT_DB.as_posix()}"


@cl.data_layer
def data_layer():
    return SQLAlchemyDataLayer(conninfo=_history_conninfo())


@cl.password_auth_callback
def auth_user(username: str, password: str) -> cl.User | None:
    """Demo credentials gate the persisted history (Chainlit requires an
    authenticated user to attribute threads to). Configure via CHAT_USER /
    CHAT_PASSWORD; defaults are demo / demo for reviewers."""
    if username == os.getenv("CHAT_USER", "demo") and password == os.getenv("CHAT_PASSWORD", "demo"):
        return cl.User(identifier=username, metadata={"role": "patient"})
    return None


# ------------------------------------------------------------------- the chat

WELCOME = (
    "**Welcome to Al-Mashreq Medical Group | أهلاً بكم في مجموعة المشرق الطبية** 🏥\n\n"
    "Describe your symptoms in Arabic or English, ask about our branches, "
    "specializations and doctors, or book an appointment.\n\n"
    "صِف الأعراض التي تشعر بها بالعربية أو الإنجليزية، أو اسأل عن فروعنا "
    "وتخصصاتنا وأطبائنا، أو احجز موعداً. يمكنك أيضاً استخدام الميكروفون 🎤"
)


def _graph_thread_id() -> str:
    """One id ties everything together: the Chainlit thread id doubles as the
    LangGraph thread id, so resuming a chat from the sidebar automatically
    restores the graph's conversation memory for it."""
    try:
        thread_id = cl.context.session.thread_id
    except Exception:
        thread_id = None
    return thread_id or str(uuid.uuid4())


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", _graph_thread_id())
    cl.user_session.set("audio_buffer", None)
    await cl.Message(content=WELCOME).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    # Chat transcript is restored by Chainlit from data/chainlit.db; pointing
    # the session at the same thread id restores the LangGraph memory too.
    cl.user_session.set("thread_id", thread["id"])
    cl.user_session.set("audio_buffer", None)


async def respond(user_text: str):
    thread_id = cl.user_session.get("thread_id")
    response = await cl.make_async(chat_turn)(graph, thread_id, user_text)

    if "action" in response:
        # Structured action: human-readable confirmation on top (derived
        # deterministically from the verified payload), raw JSON underneath.
        text = humanize(response, detect_language(user_text)) or "Structured response:"
        payload = json.dumps(response, ensure_ascii=False, indent=2)
        await cl.Message(
            content=text,
            elements=[cl.Text(name="payload.json", content=payload, language="json", display="inline")],
        ).send()
    else:
        await cl.Message(content=response.get("answer", "")).send()


async def handle_audio_bytes(audio_bytes: bytes, hint: str = "audio"):
    from app.stt import TranscriptionUnavailable, transcribe

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        path = tmp.name
    try:
        result = await cl.make_async(transcribe)(path)
    except TranscriptionUnavailable as exc:
        await cl.Message(content=f"⚠️ Voice input is not available: {exc}").send()
        return
    except Exception as exc:
        await cl.Message(content=f"⚠️ Could not transcribe the {hint}: {exc}").send()
        return

    text = result["transcribed_text"]
    if not text:
        await cl.Message(content="⚠️ No speech detected in the audio.").send()
        return
    echo = json.dumps({"transcribed_text": text}, ensure_ascii=False, indent=2)
    await cl.Message(content=f"```json\n{echo}\n```", author="transcription").send()
    await respond(text)


@cl.on_message
async def on_message(message: cl.Message):
    # Audio file attached to the message -> transcribe it first.
    audio_elements = [
        e for e in (message.elements or [])
        if (e.mime or "").startswith("audio") or str(e.path or "").lower().endswith((".wav", ".mp3", ".m4a", ".ogg", ".webm"))
    ]
    for element in audio_elements:
        with open(element.path, "rb") as f:
            await handle_audio_bytes(f.read(), hint="attached file")
    if message.content and message.content.strip():
        await respond(message.content.strip())


# --- Microphone streaming (Chainlit sends raw PCM16 chunks) ---

def _mic_sample_rate(default: int = 24000) -> int:
    """Sample rate Chainlit records the microphone at. The location of this
    setting has moved across Chainlit versions (it used to be reachable via the
    config module, now it lives on the config *singleton* at
    config.features.audio.sample_rate). Read it defensively and fall back to
    Chainlit's own default so a version bump can't break voice input again."""
    try:
        from chainlit.config import config as _cl_config
        return int(_cl_config.features.audio.sample_rate)
    except Exception:
        return default


@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_buffer", io.BytesIO())
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    buffer: io.BytesIO | None = cl.user_session.get("audio_buffer")
    if buffer is not None:
        buffer.write(chunk.data)


@cl.on_audio_end
async def on_audio_end():
    buffer: io.BytesIO | None = cl.user_session.get("audio_buffer")
    cl.user_session.set("audio_buffer", None)
    if buffer is None or buffer.getbuffer().nbytes == 0:
        return
    sample_rate = _mic_sample_rate()
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # PCM16
        wav.setframerate(sample_rate)
        wav.writeframes(buffer.getvalue())
    await handle_audio_bytes(wav_io.getvalue(), hint="recording")
