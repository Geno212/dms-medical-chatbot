"""Conversation state carried through the LangGraph workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class ChatState(TypedDict, total=False):
    # Full conversation transcript; the reducer appends new messages so the
    # checkpointer accumulates history across turns for one thread.
    messages: Annotated[list[dict[str, str]], operator.add]

    # Per-turn routing metadata (overwritten every turn).
    language: str  # "ar" | "en"
    intent: str    # "medical" | "action" | "other"
    action: str    # "book" | "list_doctors" | "list_specializations" | "list_branches" | "list_bookings" | "cancel_booking" | "none"

    # Clinical context that outlives a single turn: what the patient described
    # and which specialty the knowledge base pointed to. This is what lets
    # "yes, book me an appointment" work without the user repeating anything.
    clinical: dict[str, Any]

    # The structured response for the current turn:
    # {"answer": "..."} or {"action": ..., ...}
    response: dict[str, Any]
