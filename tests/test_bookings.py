"""Booking lifecycle: appointments are real database records — created on a
verified booking, listable, and cancellable, scoped to the conversation."""

import uuid

from conftest import FakeLLM

from app.graph.build import build_graph, chat_turn
from app.presenter import humanize


def run_graph(repo, llm):
    graph = build_graph(repo=repo, llm=llm)
    thread_id = "test-" + str(uuid.uuid4())
    return graph, thread_id


def _book(repo, thread_extra=""):
    llm = FakeLLM(
        router={"intent": "action", "action": "book"},
        slots={"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": "Cairo"},
    )
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "Book me with Dr. Sarah in Neurology at Cairo")
    return graph, thread, response


def test_booking_creates_persistent_appointment(repo):
    _, thread, response = _book(repo)
    assert response["action"] == "book_appointment"
    assert response["appointment_id"].startswith("APT-")
    assert response["status"] == "confirmed"
    saved = repo.list_appointments(thread)
    assert len(saved) == 1
    assert saved[0]["id"] == response["appointment_id"]
    assert saved[0]["doctor_id"] == "DOC-001"
    assert saved[0]["status"] == "confirmed"


def test_list_bookings_returns_saved_appointments(repo):
    llm = FakeLLM(
        router=[
            {"intent": "action", "action": "book"},
            {"intent": "action", "action": "list_bookings"},
        ],
        slots={"doctor_name": "Dr. Sarah", "specialty": None, "branch": None},
    )
    graph, thread = run_graph(repo, llm)
    booked = chat_turn(graph, thread, "Book me with Dr. Sarah")
    listed = chat_turn(graph, thread, "list my bookings")
    assert listed["action"] == "list_bookings"
    assert len(listed["appointments"]) == 1
    entry = listed["appointments"][0]
    assert entry["appointment_id"] == booked["appointment_id"]
    assert entry["doctor"] == "Dr. Sarah Hassan"
    assert entry["status"] == "confirmed"


def test_list_bookings_empty_is_a_friendly_answer(repo):
    llm = FakeLLM(router={"intent": "action", "action": "list_bookings"}, slots={})
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "what are my bookings?")
    assert "answer" in response and "action" not in response


def test_cancel_single_booking_without_reference(repo):
    llm = FakeLLM(
        router=[
            {"intent": "action", "action": "book"},
            {"intent": "action", "action": "cancel_booking"},
        ],
        slots={"doctor_name": "Dr. Sarah", "specialty": None, "branch": None},
    )
    graph, thread = run_graph(repo, llm)
    booked = chat_turn(graph, thread, "Book me with Dr. Sarah")
    cancelled = chat_turn(graph, thread, "cancel my appointment please")
    assert cancelled["action"] == "cancel_booking"
    assert cancelled["appointment_id"] == booked["appointment_id"]
    assert cancelled["status"] == "cancelled"
    assert repo.get_appointment(booked["appointment_id"])["status"] == "cancelled"


def test_cancel_among_many_requires_then_accepts_reference(repo):
    llm = FakeLLM(
        router=[
            {"intent": "action", "action": "book"},
            {"intent": "action", "action": "book"},
            {"intent": "action", "action": "cancel_booking"},
            {"intent": "action", "action": "cancel_booking"},
        ],
        slots=[
            {"doctor_name": "Dr. Sarah", "specialty": None, "branch": None},
            {"doctor_name": "Dr. Khalid", "specialty": None, "branch": None},
        ],
    )
    graph, thread = run_graph(repo, llm)
    first = chat_turn(graph, thread, "Book me with Dr. Sarah")
    chat_turn(graph, thread, "also book me with Dr. Khalid")
    ambiguous = chat_turn(graph, thread, "cancel my booking")
    assert "answer" in ambiguous  # two active bookings -> must ask, never guess
    resolved = chat_turn(graph, thread, f"cancel {first['appointment_id']}")
    assert resolved["action"] == "cancel_booking"
    assert resolved["appointment_id"] == first["appointment_id"]


def test_cancel_with_unknown_reference_fails_safely(repo):
    llm = FakeLLM(router={"intent": "action", "action": "cancel_booking"}, slots={})
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "cancel APT-000000")
    assert "answer" in response and "action" not in response


def test_heuristic_routing_for_booking_records(repo):
    from app.graph.nodes import ChatbotEngine

    assert ChatbotEngine._heuristic_route("cancel my appointment") == ("action", "cancel_booking")
    assert ChatbotEngine._heuristic_route("show my bookings please") == ("action", "list_bookings")
    assert ChatbotEngine._heuristic_route("الغي الحجز من فضلك") == ("action", "cancel_booking")
    assert ChatbotEngine._heuristic_route("اعرض حجوزاتي") == ("action", "list_bookings")
    assert ChatbotEngine._heuristic_route("I want to book an appointment") == ("action", "book")
    # editing an existing booking is its own action, not a generic "book"
    assert ChatbotEngine._heuristic_route("edit my appointment to Cairo") == ("action", "modify_booking")
    assert ChatbotEngine._heuristic_route("change my booking to another branch") == ("action", "modify_booking")
    assert ChatbotEngine._heuristic_route("عايز أعدّل موعدي") == ("action", "modify_booking")


def test_modify_booking_cancels_and_invites_rebook_no_hallucination(repo):
    """Editing a booking in place is unsupported. The bot must NOT fabricate a
    reschedule: it cancels the existing booking deterministically and asks the
    user to rebook. (Regression for the live 'edit to Cairo branch' bug where
    the medical LLM invented a fake '[Date and Time]' confirmation.)"""
    llm = FakeLLM(
        router=[
            {"intent": "action", "action": "book"},
            {"intent": "action", "action": "modify_booking"},
        ],
        slots={"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": "Cairo"},
    )
    graph, thread = run_graph(repo, llm)
    booked = chat_turn(graph, thread, "Book me with Dr. Sarah in Neurology at Cairo")
    modified = chat_turn(graph, thread, "edit my booking to the Alexandria branch")
    # A clarification-style answer, never a fabricated booking payload.
    assert "answer" in modified and "action" not in modified
    assert booked["appointment_id"] in modified["answer"]
    # The old booking is really cancelled in the database.
    assert repo.get_appointment(booked["appointment_id"])["status"] == "cancelled"


def test_modify_booking_among_many_requires_reference(repo):
    llm = FakeLLM(
        router=[
            {"intent": "action", "action": "book"},
            {"intent": "action", "action": "book"},
            {"intent": "action", "action": "modify_booking"},
        ],
        slots=[
            {"doctor_name": "Dr. Sarah", "specialty": None, "branch": None},
            {"doctor_name": "Dr. Khalid", "specialty": None, "branch": None},
        ],
    )
    graph, thread = run_graph(repo, llm)
    first = chat_turn(graph, thread, "Book me with Dr. Sarah")
    chat_turn(graph, thread, "also book me with Dr. Khalid")
    ambiguous = chat_turn(graph, thread, "edit my booking")
    assert "answer" in ambiguous  # two active -> must ask, never touch one at random
    # Neither booking was cancelled by the ambiguous request.
    assert repo.get_appointment(first["appointment_id"])["status"] == "confirmed"


def test_booking_payload_carries_arabic_fields(repo):
    """Regression: the Arabic booking confirmation showed English specialty and
    hospital because the payload had no Arabic equivalents."""
    _, _, response = _book(repo)
    assert response.get("specialty_ar")
    assert response.get("hospital_ar")
    assert response.get("branch_ar")
    arabic = humanize(response, "ar")
    # The Arabic render must use the Arabic specialty/location, not English.
    assert response["specialty_ar"] in arabic
    assert response["hospital_ar"] in arabic


def test_presenter_renders_booking_bilingually(repo):
    _, _, response = _book(repo)
    english = humanize(response, "en")
    arabic = humanize(response, "ar")
    assert response["appointment_id"] in english and "booked" in english
    assert response["appointment_id"] in arabic and "تم حجز" in arabic
    # Plain conversational answers never get a presenter overlay.
    assert humanize({"answer": "hello"}) is None
