"""Terminal chat client — the quickest way to demo the bot.

Usage:
    python -m app.cli
Commands inside the chat:
    /voice <path-to-audio-file>   transcribe an audio file and send it as a message
    /exit                         quit
"""

from __future__ import annotations

import json
import sys
import uuid

from .graph.build import build_graph, chat_turn


def print_response(response: dict) -> None:
    print("\nAssistant:")
    print(json.dumps(response, ensure_ascii=False, indent=2))
    print()


def main() -> None:
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    print("Al-Mashreq Medical Group chatbot — type in Arabic or English. /exit to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input.lower() in ("/exit", "/quit"):
            break

        if user_input.startswith("/voice "):
            from .stt import TranscriptionUnavailable, transcribe
            try:
                result = transcribe(user_input.removeprefix("/voice ").strip().strip('"'))
            except TranscriptionUnavailable as exc:
                print(f"\n[voice unavailable] {exc}\n")
                continue
            except Exception as exc:
                print(f"\n[transcription failed] {exc}\n")
                continue
            print(json.dumps({"transcribed_text": result["transcribed_text"]}, ensure_ascii=False, indent=2))
            user_input = result["transcribed_text"]
            if not user_input:
                print("[no speech detected]\n")
                continue

        print_response(chat_turn(graph, thread_id, user_input))


if __name__ == "__main__":
    main()
