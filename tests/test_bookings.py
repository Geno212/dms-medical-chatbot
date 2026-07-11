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
    """Full booking now takes two turns: request -> confirm. The first turn
    offers the resolved doctor for confirmation; the "yes" turn commits it.
    The confirm turn is handled by the pending-aware router without an LLM
    call, so it does NOT consume a router-script entry."""
    llm = FakeLLM(
        router={"intent": "action", "action": "book"},
        slots={"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": "Cairo"},
    )
    graph, thread = run_graph(repo, llm)
    offer = chat_turn(graph, thread, "Book me with Dr. Sarah in Neurology at Cairo")
    assert "answer" in offer and "action" not in offer  # confirmation prompt, no write yet
    response = chat_turn(graph, thread, "yes")
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
            {"intent": "action", "action": "book"},        # request (confirm "yes" needs no entry)
            {"intent": "action", "action": "list_bookings"},
        ],
        # specialty pins the (now duplicated) "Dr. Sarah" name to one person.
        slots={"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": None},
    )
    graph, thread = run_graph(repo, llm)
    chat_turn(graph, thread, "Book me with Dr. Sarah in Neurology")
    booked = chat_turn(graph, thread, "yes")
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
            {"intent": "action", "action": "book"},        # request (confirm "yes" needs no entry)
            {"intent": "action", "action": "cancel_booking"},
        ],
        slots={"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": None},
    )
    graph, thread = run_graph(repo, llm)
    chat_turn(graph, thread, "Book me with Dr. Sarah in Neurology")
    booked = chat_turn(graph, thread, "yes")
    cancelled = chat_turn(graph, thread, "cancel my appointment please")
    assert cancelled["action"] == "cancel_booking"
    assert cancelled["appointment_id"] == booked["appointment_id"]
    assert cancelled["status"] == "cancelled"
    assert repo.get_appointment(booked["appointment_id"])["status"] == "cancelled"


def test_cancel_among_many_requires_then_accepts_reference(repo):
    llm = FakeLLM(
        router=[
            {"intent": "action", "action": "book"},        # request #1 (confirms need no entry)
            {"intent": "action", "action": "book"},        # request #2
            {"intent": "action", "action": "cancel_booking"},
            {"intent": "action", "action": "cancel_booking"},
        ],
        slots=[
            {"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": None},
            {"doctor_name": "Dr. Khalid", "specialty": None, "branch": None},
        ],
    )
    graph, thread = run_graph(repo, llm)
    chat_turn(graph, thread, "Book me with Dr. Sarah in Neurology")
    first = chat_turn(graph, thread, "yes")
    chat_turn(graph, thread, "also book me with Dr. Khalid")
    chat_turn(graph, thread, "yes")
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
            {"intent": "action", "action": "book"},        # request (confirm "yes" needs no entry)
            {"intent": "action", "action": "modify_booking"},
        ],
        slots={"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": "Cairo"},
    )
    graph, thread = run_graph(repo, llm)
    chat_turn(graph, thread, "Book me with Dr. Sarah in Neurology at Cairo")
    booked = chat_turn(graph, thread, "yes")
    modified = chat_turn(graph, thread, "edit my booking to the Alexandria branch")
    # A clarification-style answer, never a fabricated booking payload.
    assert "answer" in modified and "action" not in modified
    assert booked["appointment_id"] in modified["answer"]
    # The old booking is really cancelled in the database.
    assert repo.get_appointment(booked["appointment_id"])["status"] == "cancelled"


def test_modify_booking_among_many_requires_reference(repo):
    llm = FakeLLM(
        router=[
            {"intent": "action", "action": "book"},        # request #1 (confirms need no entry)
            {"intent": "action", "action": "book"},        # request #2
            {"intent": "action", "action": "modify_booking"},
        ],
        slots=[
            {"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": None},
            {"doctor_name": "Dr. Khalid", "specialty": None, "branch": None},
        ],
    )
    graph, thread = run_graph(repo, llm)
    chat_turn(graph, thread, "Book me with Dr. Sarah in Neurology")
    first = chat_turn(graph, thread, "yes")
    chat_turn(graph, thread, "also book me with Dr. Khalid")
    chat_turn(graph, thread, "yes")
    ambiguous = chat_turn(graph, thread, "edit my booking")
    assert "answer" in ambiguous  # two active -> must ask, never touch one at random
    # Neither booking was cancelled by the ambiguous request.
    assert repo.get_appointment(first["appointment_id"])["status"] == "confirmed"


def test_booking_asks_for_confirmation_before_writing(repo):
    """Req 6: the resolved doctor/specialty/branch are surfaced for the user to
    confirm; nothing is written until they say yes."""
    llm = FakeLLM(
        router={"intent": "action", "action": "book"},
        slots={"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": "Cairo"},
    )
    graph, thread = run_graph(repo, llm)
    offer = chat_turn(graph, thread, "book me with Dr. Sarah")
    assert "answer" in offer and "action" not in offer
    assert "Dr. Sarah Hassan" in offer["answer"]
    assert "Neurology" in offer["answer"] and "Cairo" in offer["answer"]
    # No appointment row exists yet — the offer did not write anything.
    assert repo.list_appointments(thread) == []
    committed = chat_turn(graph, thread, "yes")
    assert committed["action"] == "book_appointment"
    assert len(repo.list_appointments(thread)) == 1


def test_same_name_doctors_trigger_clarification_not_a_guess(repo):
    """Edge case (seeded on purpose): two real "Dr. Sarah Hassan" exist in
    different specialties. Naming her without a specialty must produce a
    clarification listing the real options — never a silent pick, and nothing
    written."""
    llm = FakeLLM(
        router={"intent": "action", "action": "book"},
        slots={"doctor_name": "Dr. Sarah", "specialty": None, "branch": None},
    )
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "book me with Dr. Sarah")
    assert "answer" in response and "action" not in response
    # Mentions both specialties so the user can disambiguate.
    assert "Neurology" in response["answer"] and "Dermatology" in response["answer"]
    assert repo.list_appointments(thread) == []  # nothing booked on an ambiguous name


def test_branch_disambiguates_same_name_same_specialty(repo):
    """Edge case: two "Dr. Layla Nasser" in Cardiology (Cairo + Riyadh). Naming
    her with a specialty is still ambiguous on branch -> the bot lists both
    branches and asks; supplying the branch resolves to one and offers it."""
    llm = FakeLLM(
        router=[
            {"intent": "action", "action": "book"},   # ambiguous request
            {"intent": "action", "action": "book"},   # user names the branch -> offer
        ],
        slots=[
            {"doctor_name": "Dr. Layla", "specialty": "Cardiology", "branch": None},
            {"doctor_name": "Dr. Layla", "specialty": "Cardiology", "branch": "Cairo"},
        ],
    )
    graph, thread = run_graph(repo, llm)
    ambiguous = chat_turn(graph, thread, "book me with Dr. Layla in cardiology")
    assert "answer" in ambiguous and "action" not in ambiguous
    # The clarification distinguishes the two by BRANCH.
    assert "Cairo" in ambiguous["answer"] and "Riyadh" in ambiguous["answer"]
    assert repo.list_appointments(thread) == []  # nothing booked while ambiguous

    offer = chat_turn(graph, thread, "the Cairo one")
    assert "answer" in offer and "action" not in offer  # now a confirmation offer
    assert "Cairo" in offer["answer"]
    committed = chat_turn(graph, thread, "yes")
    assert committed["action"] == "book_appointment"
    assert committed["doctor_id"] == "DOC-022"  # the Cairo Layla
    assert committed["branch"] == "Cairo"


def test_confirmation_accepts_varied_affirmatives(repo):
    """Regression: 'exactly' (and other natural yeses) must confirm the pending
    booking, not loop back to the same offer. The LLM understands the reply;
    the heuristic is only a fallback."""
    for affirm in ["exactly", "yes please", "go ahead", "that's the one", "صح كده"]:
        llm = FakeLLM(
            router={"intent": "action", "action": "book"},
            slots={"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": None},
        )
        graph, thread = run_graph(repo, llm)
        offer = chat_turn(graph, thread, "book me with Dr. Sarah in Neurology")
        assert "answer" in offer and "action" not in offer
        committed = chat_turn(graph, thread, affirm)
        assert committed.get("action") == "book_appointment", f"{affirm!r} should confirm"


def test_unclear_reply_reasks_without_relisting(repo):
    """An ambiguous reply to a confirmation must re-ask yes/no, not silently
    re-offer (which looked stuck to the user)."""
    llm = FakeLLM(
        router={"intent": "action", "action": "book"},
        slots={"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": None},
    )
    graph, thread = run_graph(repo, llm)
    chat_turn(graph, thread, "book me with Dr. Sarah in Neurology")
    reask = chat_turn(graph, thread, "hmm")
    assert "answer" in reask and "action" not in reask
    assert "yes or no" in reask["answer"].lower() or "نعم" in reask["answer"]
    assert repo.list_appointments(thread) == []  # still nothing written


def test_declining_confirmation_writes_nothing(repo):
    llm = FakeLLM(
        router={"intent": "action", "action": "book"},
        slots={"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": "Cairo"},
    )
    graph, thread = run_graph(repo, llm)
    chat_turn(graph, thread, "book me with Dr. Sarah")
    declined = chat_turn(graph, thread, "no")
    assert "answer" in declined and "action" not in declined
    assert repo.list_appointments(thread) == []  # nothing booked


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
