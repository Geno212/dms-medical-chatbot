"""Graph nodes: intent routing, grounded medical answers, verified actions.

Division of labor:
  * LLM  -> understand language (route intents, extract slot mentions, phrase
            empathetic grounded answers)
  * Code -> everything that must be correct (entity verification against the
            DB, structured payload construction, triage escalation)
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..config import Config, get_config
from ..db import Repository
from ..llm import ChatLLM, extract_json
from ..matching import resolve_doctor, resolve_entity
from ..vectorstore import ProtocolRetriever
from . import prompts
from .state import ChatState

_ARABIC_CHARS = re.compile(r"[؀-ۿ]")


def detect_language(text: str) -> str:
    """'ar' if the message is predominantly Arabic script, else 'en'."""
    arabic = len(_ARABIC_CHARS.findall(text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    return "ar" if arabic > latin else "en"


class ChatbotEngine:
    def __init__(
        self,
        repo: Repository,
        retriever: ProtocolRetriever,
        llm: ChatLLM,
        config: Config | None = None,
    ):
        self.repo = repo
        self.retriever = retriever
        self.llm = llm
        self.config = config or get_config()
        self.group = repo.hospital_group()

    # ------------------------------------------------------------------ utils

    def _history(self, state: ChatState) -> list[dict[str, str]]:
        return state.get("messages", [])[-self.config.history_window :]

    @staticmethod
    def _last_user_message(state: ChatState) -> str:
        for message in reversed(state.get("messages", [])):
            if message["role"] == "user":
                return message["content"]
        return ""

    @staticmethod
    def _reply(response: dict[str, Any]) -> dict[str, Any]:
        """Wrap a structured response as this turn's output + transcript entry."""
        content = response.get("answer") or json.dumps(response, ensure_ascii=False)
        return {"response": response, "messages": [{"role": "assistant", "content": content}]}

    # ----------------------------------------------------------------- router

    def router_node(self, state: ChatState) -> dict[str, Any]:
        user_message = self._last_user_message(state)
        language = detect_language(user_message)
        intent, action = "medical", "none"
        try:
            raw = self.llm.chat(prompts.ROUTER_SYSTEM, self._history(state), json_mode=True)
            parsed = extract_json(raw) or {}
            if parsed.get("intent") in ("medical", "action", "other"):
                intent = parsed["intent"]
            action = parsed.get("action") or "none"
            if action not in ("book", "list_doctors", "list_specializations", "list_branches"):
                action = "none"
        except Exception:
            intent, action = self._heuristic_route(user_message)
        if intent == "action" and action == "none":
            action = "book"  # affirmative reply to a booking offer, most common case
        return {"intent": intent, "action": action, "language": language}

    @staticmethod
    def _heuristic_route(text: str) -> tuple[str, str]:
        """Keyword fallback so the bot degrades gracefully if the LLM call fails."""
        lowered = text.lower()
        if any(w in lowered for w in ("book", "appointment", "حجز", "احجز", "موعد")):
            return "action", "book"
        if any(w in lowered for w in ("doctor", "doctors", "who are", "دكتور", "طبيب", "أطباء", "اطباء")):
            return "action", "list_doctors"
        if any(w in lowered for w in ("specialization", "specialt", "department", "تخصص", "أقسام", "اقسام")):
            return "action", "list_specializations"
        if any(w in lowered for w in ("branch", "branches", "location", "فرع", "فروع")):
            return "action", "list_branches"
        return "medical", "none"

    # ---------------------------------------------------------------- medical

    def medical_node(self, state: ChatState) -> dict[str, Any]:
        language = state.get("language", "en")
        user_message = self._last_user_message(state)
        protocols = self.retriever.search(user_message, top_k=self.config.retrieval_top_k)

        clinical = dict(state.get("clinical", {}))
        specialty_name = "the relevant specialty" if language == "en" else "التخصص المناسب"
        triage_note = ""
        if protocols:
            top = protocols[0]
            specialization = self.repo.get_specialization(top["specialization_id"])
            if specialization:
                clinical = {
                    "suggested_specialization_id": specialization["id"],
                    "symptoms": user_message,
                }
                specialty_name = specialization["name_en"] if language == "en" else specialization["name_ar"]
            if any(p["triage"] == "emergency" for p in protocols[:2]):
                triage_note = prompts.TRIAGE_EMERGENCY_NOTE

        content_key = "content_ar" if language == "ar" else "content_en"
        protocol_text = "\n".join(f"- {p[content_key]}" for p in protocols) or (
            "- (No specific protocol matched; give only general comfort-and-rest advice "
            "and recommend seeing a doctor for proper evaluation.)"
        )
        system = prompts.MEDICAL_SYSTEM.format(
            hospital_en=self.group.get("name_en", ""),
            hospital_ar=self.group.get("name_ar", ""),
            protocols=protocol_text,
            triage_note=triage_note,
            language="Arabic" if language == "ar" else "English",
            specialty=specialty_name,
        )
        try:
            answer = self.llm.chat(system, self._history(state)).strip()
        except Exception:
            answer = (
                "عذراً، لم أتمكن من معالجة سؤالك الآن. حاول مرة أخرى من فضلك."
                if language == "ar"
                else "Sorry, I couldn't process your question right now. Please try again."
            )
        return {**self._reply({"answer": answer}), "clinical": clinical}

    # ----------------------------------------------------------------- action

    def action_node(self, state: ChatState) -> dict[str, Any]:
        language = state.get("language", "en")
        action = state.get("action", "book")
        slots = self._extract_slots(state)

        branch, _ = resolve_entity(slots.get("branch"), self.repo.list_branches())
        specialization, _ = resolve_entity(slots.get("specialty"), self.repo.list_specializations())

        # Slot-filling from conversation memory: if the user never named a
        # specialty but the medical discussion pointed to one, carry it over.
        if specialization is None and state.get("clinical", {}).get("suggested_specialization_id"):
            specialization = self.repo.get_specialization(
                state["clinical"]["suggested_specialization_id"]
            )

        if action == "list_branches":
            return self._reply(self._list_branches_response(language))
        if action == "list_specializations":
            return self._reply(self._list_specializations_response(branch, language))
        if action == "list_doctors":
            return self._reply(self._list_doctors_response(specialization, branch, language))
        return self._reply(self._booking_response(slots, specialization, branch, language))

    def _extract_slots(self, state: ChatState) -> dict[str, Any]:
        try:
            raw = self.llm.chat(prompts.EXTRACT_SYSTEM, self._history(state), json_mode=True)
            return extract_json(raw) or {}
        except Exception:
            return {}

    # ------------------------------------------------- deterministic payloads

    def _list_branches_response(self, language: str) -> dict[str, Any]:
        name_key = "name_ar" if language == "ar" else "name_en"
        address_key = "address_ar" if language == "ar" else "address_en"
        return {
            "action": "list_branches",
            "hospital": self.group.get("name_en"),
            "branches": [
                {"name": b[name_key], "address": b[address_key], "phone": b["phone"]}
                for b in self.repo.list_branches()
            ],
        }

    def _list_specializations_response(self, branch: dict | None, language: str) -> dict[str, Any]:
        name_key = "name_ar" if language == "ar" else "name_en"
        specializations = self.repo.list_specializations(branch["id"] if branch else None)
        return {
            "action": "list_specializations",
            "branch": branch[name_key] if branch else None,
            "specializations": [s[name_key] for s in specializations],
        }

    def _list_doctors_response(
        self, specialization: dict | None, branch: dict | None, language: str
    ) -> dict[str, Any]:
        name_key = "name_ar" if language == "ar" else "name_en"
        specialty_key = "specialty_ar" if language == "ar" else "specialty_en"
        doctors = self.repo.list_doctors(
            specialization["id"] if specialization else None,
            branch["id"] if branch else None,
        )
        return {
            "action": "list_doctors",
            "specialty": specialization[name_key] if specialization else None,
            "branch": branch[name_key] if branch else None,
            "doctors": [
                {"name": d[name_key], "specialty": d[specialty_key]} for d in doctors
            ],
        }

    def _booking_response(
        self,
        slots: dict[str, Any],
        specialization: dict | None,
        branch: dict | None,
        language: str,
    ) -> dict[str, Any]:
        """Verified booking payload, or a clarification question when the
        conversation doesn't yet pin down one real doctor."""
        doctors = self.repo.list_doctors()
        doctor_query = slots.get("doctor_name")

        if doctor_query:
            doctor, candidates = resolve_doctor(
                doctor_query,
                doctors,
                specialization["id"] if specialization else None,
                branch["id"] if branch else None,
            )
            if doctor:
                return self._verified_booking(doctor)
            if candidates:  # ambiguous mention — never guess a doctor
                return {"answer": self._ask_choose_doctor(candidates, language)}
            pool = self.repo.list_doctors(specialization["id"] if specialization else None)
            return {"answer": self._doctor_not_found(doctor_query, pool or doctors, language)}

        # No doctor named: offer the real options for the (inferred) specialty.
        pool = self.repo.list_doctors(
            specialization["id"] if specialization else None,
            branch["id"] if branch else None,
        )
        if specialization and pool:
            if len(pool) == 1:
                return self._verified_booking(pool[0])
            return {"answer": self._ask_choose_doctor(pool, language)}
        if language == "ar":
            return {"answer": "يسعدني مساعدتك في حجز موعد. مع أي تخصص أو طبيب تود الحجز؟ يمكنك أيضاً وصف الأعراض وسأرشح لك التخصص المناسب."}
        return {"answer": "I'd be happy to book an appointment for you. Which specialty or doctor would you like? You can also describe your symptoms and I'll suggest the right specialty."}

    def _verified_booking(self, doctor: dict[str, Any]) -> dict[str, Any]:
        branch = self.repo.get_branch(doctor["branch_id"])
        return {
            "action": "book_appointment",
            "doctor_id": doctor["id"],
            "doctor_name": doctor["name_en"],
            "doctor_name_ar": doctor["name_ar"],
            "specialty": doctor["specialty_en"],
            "specialty_id": doctor["specialization_id"],
            "branch": branch["name_en"],
            "branch_id": branch["id"],
            "hospital": self.repo.hospital_label(branch),
        }

    @staticmethod
    def _describe(doctor: dict[str, Any], language: str) -> str:
        if language == "ar":
            return f"{doctor['name_ar']} ({doctor['specialty_ar']} - فرع {doctor['branch_ar']})"
        return f"{doctor['name_en']} ({doctor['specialty_en']} - {doctor['branch_en']} branch)"

    def _ask_choose_doctor(self, doctors: list[dict[str, Any]], language: str) -> str:
        options = "؛ ".join(self._describe(d, language) for d in doctors[:6]) if language == "ar" \
            else "; ".join(self._describe(d, language) for d in doctors[:6])
        if language == "ar":
            return f"لدينا أكثر من خيار مناسب: {options}. مع مَن تودّ الحجز؟"
        return f"We have more than one suitable option: {options}. Who would you like to book with?"

    def _doctor_not_found(self, query: str, alternatives: list[dict[str, Any]], language: str) -> str:
        options = "; ".join(self._describe(d, language) for d in alternatives[:6])
        if language == "ar":
            return f"عذراً، لا يوجد لدينا طبيب باسم \"{query}\". الأطباء المتاحون لدينا: {options}. مع مَن تودّ الحجز؟"
        return f"Sorry, we don't have a doctor named \"{query}\". Our available doctors are: {options}. Who would you like to book with?"

    # ------------------------------------------------------------------ other

    def other_node(self, state: ChatState) -> dict[str, Any]:
        language = state.get("language", "en")
        branches = ", ".join(
            b["name_ar" if language == "ar" else "name_en"] for b in self.repo.list_branches()
        )
        system = prompts.OTHER_SYSTEM.format(
            hospital_en=self.group.get("name_en", ""),
            hospital_ar=self.group.get("name_ar", ""),
            branches=branches,
            language="Arabic" if language == "ar" else "English",
        )
        try:
            answer = self.llm.chat(system, self._history(state)).strip()
        except Exception:
            answer = (
                f"أهلاً بك في {self.group.get('name_ar', '')}! كيف يمكنني مساعدتك اليوم؟"
                if language == "ar"
                else f"Welcome to {self.group.get('name_en', '')}! How can I help you today?"
            )
        return self._reply({"answer": answer})
