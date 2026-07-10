"""Deterministic presentation layer for structured action payloads.

The response contract stays pure (the graph returns either conversational
text OR a structured action payload, per the task spec). The UI, however,
should not greet a patient with raw JSON: this module renders a
human-readable bilingual confirmation from the verified payload, and the
UI shows the payload itself underneath as the machine-readable artifact.

No LLM is involved: the text is derived 1:1 from already-verified data, so
it can never contradict the payload (and it costs zero latency).
"""

from __future__ import annotations

from typing import Any


def _booking_en(p: dict[str, Any]) -> str:
    return (
        f"✅ **Your appointment is booked.**\n\n"
        f"- **Doctor:** {p['doctor_name']}\n"
        f"- **Specialty:** {p['specialty']}\n"
        f"- **Location:** {p['hospital']}\n"
        f"- **Booking reference:** `{p['appointment_id']}`\n\n"
        f"You can say \"list my bookings\" or \"cancel my booking\" anytime."
    )


def _booking_ar(p: dict[str, Any]) -> str:
    return (
        f"✅ **تم حجز موعدك بنجاح.**\n\n"
        f"- **الطبيب:** {p.get('doctor_name_ar') or p['doctor_name']}\n"
        f"- **التخصص:** {p.get('specialty_ar') or p['specialty']}\n"
        f"- **المكان:** {p.get('hospital_ar') or p['hospital']}\n"
        f"- **رقم الحجز:** `{p['appointment_id']}`\n\n"
        f"يمكنك قول \"اعرض حجوزاتي\" أو \"الغِ حجزي\" في أي وقت."
    )


def _doctors_en(p: dict[str, Any]) -> str:
    scope = " and ".join(x for x in (p.get("specialty"), p.get("branch") and f"{p['branch']} branch") if x)
    intro = f"Here are our doctors{' in ' + scope if scope else ''}:"
    lines = "\n".join(f"- **{d['name']}** ({d['specialty']})" for d in p["doctors"])
    return f"{intro}\n\n{lines}" if p["doctors"] else "No doctors match that yet."


def _doctors_ar(p: dict[str, Any]) -> str:
    intro = "هؤلاء أطباؤنا المتاحون:"
    lines = "\n".join(f"- **{d['name']}** ({d['specialty']})" for d in p["doctors"])
    return f"{intro}\n\n{lines}" if p["doctors"] else "لا يوجد أطباء مطابقون حالياً."


def _specializations_en(p: dict[str, Any]) -> str:
    where = f" at our {p['branch']} branch" if p.get("branch") else ""
    return f"Specializations available{where}:\n\n" + "\n".join(f"- {s}" for s in p["specializations"])


def _specializations_ar(p: dict[str, Any]) -> str:
    where = f" في فرع {p['branch']}" if p.get("branch") else ""
    return f"التخصصات المتاحة{where}:\n\n" + "\n".join(f"- {s}" for s in p["specializations"])


def _branches_en(p: dict[str, Any]) -> str:
    lines = "\n".join(f"- **{b['name']}** — {b['address']} (☎ {b['phone']})" for b in p["branches"])
    return f"{p['hospital']} branches:\n\n{lines}"


def _branches_ar(p: dict[str, Any]) -> str:
    lines = "\n".join(f"- **{b['name']}** — {b['address']} (☎ {b['phone']})" for b in p["branches"])
    return f"فروع {p['hospital']}:\n\n{lines}"


_STATUS_AR = {"confirmed": "مؤكد", "cancelled": "ملغي"}


def _bookings_en(p: dict[str, Any]) -> str:
    lines = "\n".join(
        f"- `{a['appointment_id']}` — **{a['doctor']}** ({a['specialty']}, {a['branch']} branch) — {a['status']} — {a['created_at']}"
        for a in p["appointments"]
    )
    return f"Your bookings in this conversation:\n\n{lines}"


def _bookings_ar(p: dict[str, Any]) -> str:
    lines = "\n".join(
        f"- `{a['appointment_id']}` — **{a['doctor']}** ({a['specialty']}، فرع {a['branch']}) — {_STATUS_AR.get(a['status'], a['status'])} — {a['created_at']}"
        for a in p["appointments"]
    )
    return f"حجوزاتك في هذه المحادثة:\n\n{lines}"


def _cancel_en(p: dict[str, Any]) -> str:
    return (
        f"🗑️ **Booking cancelled.**\n\n"
        f"- **Reference:** `{p['appointment_id']}`\n"
        f"- **Doctor:** {p['doctor']} ({p['specialty']}, {p['branch']} branch)\n\n"
        f"Would you like to book a different appointment?"
    )


def _cancel_ar(p: dict[str, Any]) -> str:
    return (
        f"🗑️ **تم إلغاء الحجز.**\n\n"
        f"- **رقم الحجز:** `{p['appointment_id']}`\n"
        f"- **الطبيب:** {p['doctor']} ({p['specialty']}، فرع {p['branch']})\n\n"
        f"هل تودّ حجز موعد آخر؟"
    )


_RENDERERS = {
    "book_appointment": (_booking_en, _booking_ar),
    "list_doctors": (_doctors_en, _doctors_ar),
    "list_specializations": (_specializations_en, _specializations_ar),
    "list_branches": (_branches_en, _branches_ar),
    "list_bookings": (_bookings_en, _bookings_ar),
    "cancel_booking": (_cancel_en, _cancel_ar),
}


def humanize(response: dict[str, Any], language: str = "en") -> str | None:
    """Human-readable text for an action payload; None for plain answers
    (they are already conversational) or unknown action shapes."""
    action = response.get("action")
    if not action or action not in _RENDERERS:
        return None
    en, ar = _RENDERERS[action]
    try:
        return ar(response) if language == "ar" else en(response)
    except (KeyError, TypeError):
        return None  # never let presentation break the raw payload path
