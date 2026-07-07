"""Chainlit web UI.

Run:  chainlit run app/chainlit_app.py -w

* Text chat in Arabic or English.
* Voice: press the microphone button (streamed PCM is buffered and
  transcribed locally with faster-whisper), or attach an audio file to a
  message. The transcription is echoed as {"transcribed_text": ...} and then
  processed as a regular chat message — exactly as the task specifies.
* Medical answers render as chat text; action responses render as the raw
  structured JSON payload.
"""

from __future__ import annotations

import io
import json
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

from app.graph.build import build_graph, chat_turn

graph = build_graph()

WELCOME = (
    "**Welcome to Al-Mashreq Medical Group | أهلاً بكم في مجموعة المشرق الطبية** 🏥\n\n"
    "Describe your symptoms in Arabic or English, ask about our branches, "
    "specializations and doctors, or book an appointment.\n\n"
    "صِف الأعراض التي تشعر بها بالعربية أو الإنجليزية، أو اسأل عن فروعنا "
    "وتخصصاتنا وأطبائنا، أو احجز موعداً. يمكنك أيضاً استخدام الميكروفون 🎤"
)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("thread_id", str(uuid.uuid4()))
    cl.user_session.set("audio_buffer", None)
    await cl.Message(content=WELCOME).send()


async def respond(user_text: str):
    thread_id = cl.user_session.get("thread_id")
    response = await cl.make_async(chat_turn)(graph, thread_id, user_text)
    if "answer" in response and "action" not in response:
        await cl.Message(content=response["answer"]).send()
    else:
        payload = json.dumps(response, ensure_ascii=False, indent=2)
        await cl.Message(content=f"```json\n{payload}\n```").send()


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
    sample_rate = cl.config.features.audio.sample_rate
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # PCM16
        wav.setframerate(sample_rate)
        wav.writeframes(buffer.getvalue())
    await handle_audio_bytes(wav_io.getvalue(), hint="recording")
