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
from ..logging_setup import get_logger
from ..matching import resolve_doctor, resolve_entity
from ..vectorstore import ProtocolRetriever
from . import prompts
from .state import ChatState

_ARABIC_CHARS = re.compile(r"[؀-ۿ]")
log = get_logger()


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

    def _reply_committing(self, response: dict[str, Any]) -> dict[str, Any]:
        """A confirmed booking was written -> clear the pending offer."""
        return {**self._reply(response), "pending_booking": {}}

    def _reply_clearing(self, response: dict[str, Any]) -> dict[str, Any]:
        """A pending offer was declined/abandoned -> clear it."""
        return {**self._reply(response), "pending_booking": {}}

    _AFFIRM = ("yes", "yeah", "yep", "sure", "ok", "okay", "confirm", "please do",
               "go ahead", "نعم", "اه", "أيوه", "ايوه", "تمام", "اكيد", "أكيد",
               "احجز", "احجزلي", "اوك", "ماشي", "موافق")
    _DECLINE = ("no", "nope", "cancel", "don't", "dont", "not now", "stop",
                "لا", "لأ", "مش", "إلغاء", "الغاء", "لا شكرا", "مش عايز")

    @classmethod
    def _confirm_decision(cls, text: str) -> str:
        """Classify a reply to a booking-confirmation question as 'yes', 'no',
        or 'unclear' (a fresh request that discards the pending offer)."""
        t = (text or "").strip().lower()
        if any(w in t for w in cls._DECLINE):
            return "no"
        if any(t == w or t.startswith(w + " ") or w in t.split() for w in cls._AFFIRM):
            return "yes"
        return "unclear"

    # ----------------------------------------------------------------- router

    def router_node(self, state: ChatState) -> dict[str, Any]:
        user_message = self._last_user_message(state)
        language = detect_language(user_message)

        # If a booking is awaiting confirmation and the user gives a clear
        # yes/no, route straight to the action path so the confirmation is
        # honored — even if the router LLM is unavailable (a bare "yes" is not
        # a booking keyword the heuristic would otherwise catch).
        if (state.get("pending_booking") or {}).get("doctor_id"):
            if self._confirm_decision(user_message) in ("yes", "no"):
                log.info("router: pending booking + %r -> action/book (confirmation)", user_message[:40])
                return {"intent": "action", "action": "book", "language": language}

        intent, action = "medical", "none"
        source = "llm"
        try:
            raw = self.llm.chat(prompts.ROUTER_SYSTEM, self._history(state), json_mode=True)
            log.debug("router raw LLM output: %s", raw)
            parsed = extract_json(raw) or {}
            if parsed.get("intent") in ("medical", "action", "other"):
                intent = parsed["intent"]
            action = parsed.get("action") or "none"
            if action not in ("book", "list_doctors", "list_specializations",
                              "list_branches", "list_bookings", "cancel_booking",
                              "modify_booking"):
                action = "none"
        except Exception as exc:
            source = "heuristic-fallback"
            intent, action = self._heuristic_route(user_message)
            log.warning("router LLM call failed (%s: %s) -> falling back to keyword heuristics", type(exc).__name__, exc)
        if action != "none" and intent != "action":
            # Contradictory LLM output (e.g. intent=medical + action=list_doctors).
            # The named action is the more specific signal — trust it.
            log.info("router emitted intent=%s with action=%s -> coercing intent to 'action'", intent, action)
            intent = "action"
        if intent == "action" and action == "none":
            action = "book"  # affirmative reply to a booking offer, most common case
        log.info(
            "router: message=%r language=%s intent=%s action=%s source=%s",
            user_message[:80], language, intent, action, source,
        )
        return {"intent": intent, "action": action, "language": language}

    @staticmethod
    def _heuristic_route(text: str) -> tuple[str, str]:
        """Keyword fallback so the bot degrades gracefully if the LLM call fails."""
        lowered = text.lower()
        booking_words = ("book", "appointment", "حجز", "احجز", "موعد", "مواعيد")
        modify_words = ("edit", "change", "modify", "reschedule", "move", "عدل", "عدّل", "غير", "غيّر", "تعديل")
        if any(w in lowered for w in ("cancel", "الغاء", "إلغاء", "الغي", "ألغي")) \
                and any(w in lowered for w in booking_words):
            return "action", "cancel_booking"
        if any(w in lowered for w in modify_words) \
                and any(w in lowered for w in booking_words):
            return "action", "modify_booking"
        if any(w in lowered for w in ("my booking", "my appointment", "حجوزاتي", "مواعيدي")):
            return "action", "list_bookings"
        if any(w in lowered for w in booking_words):
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
        log.info(
            "retrieval: query=%r hits=%s",
            user_message[:80],
            [(p["id"], p["score"], p["triage"]) for p in protocols],
        )

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
                log.info("clinical context updated: suggested_specialization=%s", specialization["name_en"])
            # Escalate only when an emergency protocol matches *strongly*. A
            # weak emergency hit (e.g. a booking-management message scraping an
            # emergency protocol at score ~0.20) must not force emergency
            # guidance — that misfires triage on non-medical turns.
            emergency = next(
                (
                    p for p in protocols[:2]
                    if p["triage"] == "emergency" and p["score"] >= self.config.triage_min_score
                ),
                None,
            )
            if emergency:
                triage_note = prompts.TRIAGE_EMERGENCY_NOTE
                log.warning(
                    "triage escalation: emergency protocol %s matched for %r (score=%.4f)",
                    emergency["id"], user_message[:80], emergency["score"],
                )

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

    def action_node(self, state: ChatState, config: dict | None = None) -> dict[str, Any]:
        language = state.get("language", "en")
        action = state.get("action", "book")
        thread_id = ((config or {}).get("configurable") or {}).get("thread_id", "default")

        # Booking-record actions are fully deterministic (no slots needed).
        if action == "list_bookings":
            return self._reply(self._list_bookings_response(thread_id, language))
        if action == "cancel_booking":
            return self._reply(
                self._cancel_booking_response(thread_id, self._last_user_message(state), language)
            )
        if action == "modify_booking":
            return self._reply(
                self._modify_booking_response(thread_id, self._last_user_message(state), language)
            )

        # A booking already resolved to one doctor may be awaiting the user's
        # explicit yes/no. Resolve that first so "yes" commits it and "no"
        # cancels the offer — without re-running slot extraction on an empty
        # affirmative. Any message that names a *different* doctor/specialty
        # falls through and re-resolves (the pending offer is discarded).
        pending = state.get("pending_booking") or {}
        if pending.get("doctor_id"):
            decision = self._confirm_decision(self._last_user_message(state))
            if decision == "yes":
                doctor = self.repo.get_doctor(pending["doctor_id"])
                log.info("pending booking confirmed by user -> committing %s", pending["doctor_id"])
                return self._reply_committing(self._commit_booking(doctor, thread_id))
            if decision == "no":
                log.info("pending booking declined by user -> cleared")
                text = ("تمام، لم أُتمّ الحجز. مع مَن أو في أي تخصص تودّ الحجز؟"
                        if language == "ar"
                        else "No problem, I won't book that. Who or which specialty would you like instead?")
                return self._reply_clearing({"answer": text})
            # Not a clear yes/no -> treat as a fresh request; discard the offer
            # and continue to normal resolution below.

        slots = self._extract_slots(state)
        log.info("slot extraction: %s", slots)

        branch, branch_candidates = resolve_entity(slots.get("branch"), self.repo.list_branches())
        specialization, spec_candidates = resolve_entity(slots.get("specialty"), self.repo.list_specializations())
        log.info(
            "entity resolution: branch=%r -> %s (%d candidates) | specialty=%r -> %s (%d candidates)",
            slots.get("branch"), branch and branch["name_en"], len(branch_candidates),
            slots.get("specialty"), specialization and specialization["name_en"], len(spec_candidates),
        )

        # Slot-filling from conversation memory: if the user never named a
        # specialty but the medical discussion pointed to one, carry it over.
        if specialization is None and state.get("clinical", {}).get("suggested_specialization_id"):
            specialization = self.repo.get_specialization(
                state["clinical"]["suggested_specialization_id"]
            )
            log.info("specialty carried over from clinical context: %s", specialization and specialization["name_en"])

        if action == "list_branches":
            return self._reply(self._list_branches_response(language))
        if action == "list_specializations":
            return self._reply(self._list_specializations_response(branch, language))
        if action == "list_doctors":
            return self._reply(self._list_doctors_response(specialization, branch, language))
        response, pending = self._booking_response(slots, specialization, branch, language, thread_id)
        return {**self._reply(response), "pending_booking": pending}

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
        thread_id: str = "default",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Returns (response, pending_booking). When the conversation pins down
        exactly one real doctor, the response is a confirmation question and
        pending_booking holds that doctor (committed on the user's next "yes").
        Otherwise it's a clarification question and pending_booking is empty."""
        doctors = self.repo.list_doctors()
        doctor_query = slots.get("doctor_name")

        # The extractor occasionally mislabels a specialty mention as a doctor
        # name (e.g. the user echoes the assistant's "our Neurology doctors"
        # and the LLM emits doctor_name="Neurology"). A specialization is
        # never a person: reroute it to the specialty slot instead of failing
        # the doctor lookup with "no doctor named Neurology".
        if doctor_query:
            as_specialization, _ = resolve_entity(
                doctor_query, self.repo.list_specializations()
            )
            if as_specialization:
                log.info(
                    "doctor_name %r resolved as a specialty (%s), not a person -> rerouted to specialty slot",
                    doctor_query, as_specialization["name_en"],
                )
                specialization = specialization or as_specialization
                doctor_query = None

        if doctor_query:
            doctor, candidates = resolve_doctor(
                doctor_query,
                doctors,
                specialization["id"] if specialization else None,
                branch["id"] if branch else None,
            )
            if doctor:
                log.info("doctor resolved: %r -> %s (%s)", doctor_query, doctor["name_en"], doctor["id"])
                return self._offer_booking(doctor, language)
            if candidates:  # ambiguous mention — never guess a doctor
                log.info("doctor mention ambiguous: %r -> %d candidates, asking user to clarify", doctor_query, len(candidates))
                return {"answer": self._ask_choose_doctor(candidates, language)}, {}
            pool = self.repo.list_doctors(specialization["id"] if specialization else None)
            log.info("doctor not found: %r has no match in database", doctor_query)
            return {"answer": self._doctor_not_found(doctor_query, pool or doctors, language)}, {}

        # No doctor named: offer the real options for the (inferred) specialty.
        pool = self.repo.list_doctors(
            specialization["id"] if specialization else None,
            branch["id"] if branch else None,
        )
        if specialization and pool:
            if len(pool) == 1:
                log.info("single doctor matches specialty %s -> resolved: %s", specialization["name_en"], pool[0]["name_en"])
                return self._offer_booking(pool[0], language)
            log.info("%d doctors match specialty %s -> asking user to choose", len(pool), specialization["name_en"])
            return {"answer": self._ask_choose_doctor(pool, language)}, {}
        if language == "ar":
            return {"answer": "يسعدني مساعدتك في حجز موعد. مع أي تخصص أو طبيب تود الحجز؟ يمكنك أيضاً وصف الأعراض وسأرشح لك التخصص المناسب."}, {}
        return {"answer": "I'd be happy to book an appointment for you. Which specialty or doctor would you like? You can also describe your symptoms and I'll suggest the right specialty."}, {}

    def _offer_booking(
        self, doctor: dict[str, Any], language: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Ask the user to confirm the fully-resolved doctor/specialty/branch
        before anything is written. Nothing hits the database yet; the doctor
        is stashed in pending_booking so the next 'yes' commits it. This is
        what makes the identified doctor, specialty, and branch explicit to the
        user (task requirement 6) instead of booking silently."""
        branch = self.repo.get_branch(doctor["branch_id"])
        if language == "ar":
            answer = (
                f"سأحجز لك مع {doctor['name_ar']} "
                f"(تخصص {doctor['specialty_ar']}، فرع {branch['name_ar']}). "
                f"هل أؤكّد الحجز؟"
            )
        else:
            answer = (
                f"I'll book you with {doctor['name_en']} "
                f"({doctor['specialty_en']}, {branch['name_en']} branch). "
                f"Shall I confirm this booking?"
            )
        log.info("offering booking for confirmation: %s (%s)", doctor["name_en"], doctor["id"])
        return {"answer": answer}, {"doctor_id": doctor["id"]}

    def _commit_booking(self, doctor: dict[str, Any], thread_id: str) -> dict[str, Any]:
        """Write the confirmed appointment and build the verified payload from
        database rows only (task requirement 6: doctor/specialty/branch/hospital
        identifiers, all from real data — the model never produces field values)."""
        branch = self.repo.get_branch(doctor["branch_id"])
        appointment = self.repo.create_appointment(thread_id, doctor["id"])
        log.info("appointment saved: %s (%s, thread=%s)", appointment["id"], doctor["name_en"], thread_id)
        return {
            "action": "book_appointment",
            "appointment_id": appointment["id"],
            "status": appointment["status"],
            "created_at": appointment["created_at"],
            "doctor_id": doctor["id"],
            "doctor_name": doctor["name_en"],
            "doctor_name_ar": doctor["name_ar"],
            "specialty": doctor["specialty_en"],
            "specialty_ar": doctor["specialty_ar"],
            "specialty_id": doctor["specialization_id"],
            "branch": branch["name_en"],
            "branch_ar": branch["name_ar"],
            "branch_id": branch["id"],
            "hospital": self.repo.hospital_label(branch),
            "hospital_ar": self.repo.hospital_label(branch, language="ar"),
        }

    # -------------------------------------------- booking records (CRUD-lite)

    _APPOINTMENT_ID = re.compile(r"APT-[0-9A-Fa-f]{6}")

    def _appointment_view(self, a: dict[str, Any], language: str) -> dict[str, Any]:
        ar = language == "ar"
        return {
            "appointment_id": a["id"],
            "status": a["status"],
            "doctor": a["doctor_ar"] if ar else a["doctor_en"],
            "specialty": a["specialty_ar"] if ar else a["specialty_en"],
            "branch": a["branch_ar"] if ar else a["branch_en"],
            "created_at": a["created_at"],
        }

    def _list_bookings_response(self, thread_id: str, language: str) -> dict[str, Any]:
        appointments = self.repo.list_appointments(thread_id)
        log.info("list bookings: thread=%s -> %d record(s)", thread_id, len(appointments))
        if not appointments:
            if language == "ar":
                return {"answer": "لا توجد لديك حجوزات حتى الآن في هذه المحادثة. هل تودّ حجز موعد؟ يمكنك وصف الأعراض وسأرشح لك التخصص المناسب."}
            return {"answer": "You don't have any bookings yet in this conversation. Would you like to book an appointment? You can describe your symptoms and I'll suggest the right specialty."}
        return {
            "action": "list_bookings",
            "appointments": [self._appointment_view(a, language) for a in appointments],
        }

    def _cancel_booking_response(self, thread_id: str, user_message: str, language: str) -> dict[str, Any]:
        appointments = self.repo.list_appointments(thread_id)
        active = [a for a in appointments if a["status"] == "confirmed"]
        mentioned = self._APPOINTMENT_ID.search(user_message or "")

        target = None
        if mentioned:
            wanted = mentioned.group(0).upper()
            target = next((a for a in active if a["id"] == wanted), None)
            if target is None:
                if language == "ar":
                    return {"answer": f"لم أجد حجزاً نشطاً بالرقم {wanted} في هذه المحادثة."}
                return {"answer": f"I couldn't find an active booking with reference {wanted} in this conversation."}
        elif len(active) == 1:
            target = active[0]
        elif not active:
            if language == "ar":
                return {"answer": "لا توجد لديك حجوزات نشطة لإلغائها."}
            return {"answer": "You have no active bookings to cancel."}
        else:
            options = "؛ ".join(f"{a['id']} ({a['doctor_ar']})" for a in active) if language == "ar" \
                else "; ".join(f"{a['id']} ({a['doctor_en']})" for a in active)
            if language == "ar":
                return {"answer": f"لديك أكثر من حجز نشط: {options}. أي حجز تودّ إلغاءه؟ اذكر رقم الحجز."}
            return {"answer": f"You have more than one active booking: {options}. Which one would you like to cancel? Please give the booking reference."}

        cancelled = self.repo.cancel_appointment(target["id"])
        log.info("appointment cancelled: %s (thread=%s)", target["id"], thread_id)
        return {
            "action": "cancel_booking",
            **self._appointment_view(cancelled, language),
        }

    def _modify_booking_response(self, thread_id: str, user_message: str, language: str) -> dict[str, Any]:
        """Editing a booking in place (change branch/doctor/time) is not a
        supported operation — there is no reschedule path in the data layer.
        Rather than let the model hallucinate a fake reschedule, we do the
        honest, deterministic thing: cancel the existing booking and invite the
        user to book again. Reuses the same never-guess target selection as
        cancellation (an APT reference, or the single active booking, or a
        clarification prompt when several are active)."""
        appointments = self.repo.list_appointments(thread_id)
        active = [a for a in appointments if a["status"] == "confirmed"]
        mentioned = self._APPOINTMENT_ID.search(user_message or "")

        if not active:
            if language == "ar":
                return {"answer": "لا يوجد لديك حجز نشط لتعديله. هل تودّ حجز موعد جديد؟"}
            return {"answer": "You don't have an active booking to change. Would you like to make a new booking?"}

        target = None
        if mentioned:
            wanted = mentioned.group(0).upper()
            target = next((a for a in active if a["id"] == wanted), None)
            if target is None:
                if language == "ar":
                    return {"answer": f"لم أجد حجزاً نشطاً بالرقم {wanted} في هذه المحادثة."}
                return {"answer": f"I couldn't find an active booking with reference {wanted} in this conversation."}
        elif len(active) == 1:
            target = active[0]
        else:
            options = "؛ ".join(f"{a['id']} ({a['doctor_ar']})" for a in active) if language == "ar" \
                else "; ".join(f"{a['id']} ({a['doctor_en']})" for a in active)
            if language == "ar":
                return {"answer": f"لديك أكثر من حجز نشط: {options}. أي حجز تودّ تعديله؟ اذكر رقم الحجز."}
            return {"answer": f"You have more than one active booking: {options}. Which one would you like to change? Please give the booking reference."}

        cancelled = self.repo.cancel_appointment(target["id"])
        log.info("booking modify -> cancelled %s for rebooking (thread=%s)", target["id"], thread_id)
        ref = target["id"]
        if language == "ar":
            doctor = target["doctor_ar"]
            return {"answer": (
                f"لا يمكنني تعديل حجز قائم في مكانه. لقد ألغيت حجزك الحالي "
                f"({ref} مع {doctor})، ويمكنك الآن الحجز من جديد بالتخصص أو الطبيب أو الفرع الذي تريده. "
                f"مع مَن أو في أي فرع تودّ الحجز؟"
            )}
        doctor = target["doctor_en"]
        return {"answer": (
            f"I can't change a booking in place. I've cancelled your current booking "
            f"({ref} with {doctor}), and you can book again with whatever doctor, "
            f"specialty, or branch you'd like. Who or which branch would you like to book with?"
        )}

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
