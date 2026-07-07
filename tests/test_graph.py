"""End-to-end graph tests: routing, RAG grounding, verified actions, and the
multi-turn slot-filling flow — all with a scripted fake LLM."""

import uuid

from conftest import FakeLLM

from app.graph.build import build_graph, chat_turn
from app.graph.nodes import detect_language


def run_graph(repo, llm):
    graph = build_graph(repo=repo, llm=llm)
    thread_id = "test-" + str(uuid.uuid4())  # prefix lets conftest scrub booking rows
    return graph, thread_id


def test_detect_language():
    assert detect_language("I have a headache") == "en"
    assert detect_language("عندي صداع شديد") == "ar"
    assert detect_language("أريد حجز موعد مع Dr. Sarah") == "ar"


def test_medical_question_returns_answer_json(repo):
    llm = FakeLLM(router={"intent": "medical", "action": "none"})
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "I've been having a severe headache and fever for two days")
    assert set(response.keys()) == {"answer"}
    assert isinstance(response["answer"], str) and response["answer"]


def test_full_booking_request_returns_verified_payload(repo):
    llm = FakeLLM(
        router={"intent": "action", "action": "book"},
        slots={"doctor_name": "Dr. Sarah", "specialty": "Neurology", "branch": "Cairo"},
    )
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "I want to book with Dr. Sarah in Neurology at the Cairo branch")
    assert response["action"] == "book_appointment"
    assert response["doctor_id"] == "DOC-001"
    assert response["doctor_name"] == "Dr. Sarah Hassan"
    assert response["specialty"] == "Neurology"
    assert response["branch"] == "Cairo"
    assert response["hospital"] == "Al-Mashreq Medical Group - Cairo Branch"


def test_multi_turn_medical_to_booking_carries_specialty(repo):
    """The PDF's core scenario: symptoms -> recommendation -> 'yes, book me'
    with no entities repeated. The clinical context must fill the specialty."""
    llm = FakeLLM(
        router=[
            {"intent": "medical", "action": "none"},
            {"intent": "action", "action": "book"},
        ],
        slots={"doctor_name": None, "specialty": None, "branch": None},
    )
    graph, thread = run_graph(repo, llm)
    first = chat_turn(graph, thread, "I've had a severe headache and fever for two days")
    assert "answer" in first
    second = chat_turn(graph, thread, "yes, please book me an appointment")
    # Two neurologists exist (Cairo + Alexandria) -> clarification, not a guess.
    assert "answer" in second
    assert "Dr. Sarah Hassan" in second["answer"]
    assert "Dr. Omar El-Sayed" in second["answer"]


def test_multi_turn_booking_resolves_after_choice(repo):
    llm = FakeLLM(
        router=[
            {"intent": "medical", "action": "none"},
            {"intent": "action", "action": "book"},
            {"intent": "action", "action": "book"},
        ],
        slots=[
            {"doctor_name": None, "specialty": None, "branch": None},
            {"doctor_name": "Dr. Sarah", "specialty": None, "branch": None},
        ],
    )
    graph, thread = run_graph(repo, llm)
    chat_turn(graph, thread, "I've had a severe headache and fever for two days")
    chat_turn(graph, thread, "yes, book me an appointment")
    final = chat_turn(graph, thread, "Dr. Sarah please")
    assert final["action"] == "book_appointment"
    assert final["doctor_id"] == "DOC-001"
    assert final["specialty"] == "Neurology"  # carried from the medical turn


def test_specialty_mislabeled_as_doctor_name_is_not_rejected(repo):
    """Regression: after the bot offered "our Neurology doctors", the user said
    "yes, book me" and the extractor emitted doctor_name="Neurology". The bot
    must recognize that as a specialty and offer its real doctors — not reply
    'we don't have a doctor named "Neurology"'."""
    llm = FakeLLM(
        router=[
            {"intent": "medical", "action": "none"},
            {"intent": "action", "action": "book"},
        ],
        slots={"doctor_name": "Neurology", "specialty": None, "branch": None},
    )
    graph, thread = run_graph(repo, llm)
    chat_turn(graph, thread, "I've had a severe headache and fever for two days")
    response = chat_turn(graph, thread, "Yes, book me an appointment")
    assert "answer" in response
    assert "don't have a doctor" not in response["answer"]
    assert "Dr. Sarah Hassan" in response["answer"]
    assert "Dr. Omar El-Sayed" in response["answer"]


def test_booking_with_specialty_as_doctor_name_single_turn(repo):
    llm = FakeLLM(
        router={"intent": "action", "action": "book"},
        slots={"doctor_name": "Neurology", "specialty": "Neurology", "branch": None},
    )
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "Book me an appointment with Neurology")
    assert "answer" in response
    assert "don't have a doctor" not in response["answer"]
    assert "Dr. Sarah Hassan" in response["answer"]
    assert "Dr. Omar El-Sayed" in response["answer"]


def test_contradictory_router_output_trusts_the_specific_action(repo):
    """Regression: the router LLM sometimes emits intent=medical together with
    a concrete action (e.g. list_doctors). The named action is the more
    specific signal — the turn must produce the structured lookup, not a
    medical RAG answer."""
    llm = FakeLLM(
        router={"intent": "medical", "action": "list_doctors"},
        slots={"doctor_name": None, "specialty": "cardiologists", "branch": "Riyadh"},
    )
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "Who are the cardiologists at the Riyadh branch?")
    assert response["action"] == "list_doctors"
    assert {d["name"] for d in response["doctors"]} == {"Dr. Khalid Al-Mansouri", "Dr. Layla Nasser"}


def test_list_doctors_cardiology_riyadh(repo):
    llm = FakeLLM(
        router={"intent": "action", "action": "list_doctors"},
        slots={"doctor_name": None, "specialty": "cardiologists", "branch": "Riyadh"},
    )
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "Who are the cardiologists at the Riyadh branch?")
    assert response["action"] == "list_doctors"
    assert response["specialty"] == "Cardiology"
    assert response["branch"] == "Riyadh"
    names = {d["name"] for d in response["doctors"]}
    assert names == {"Dr. Khalid Al-Mansouri", "Dr. Layla Nasser"}


def test_list_doctors_arabic_returns_arabic_names(repo):
    llm = FakeLLM(
        router={"intent": "action", "action": "list_doctors"},
        slots={"doctor_name": None, "specialty": "قلب", "branch": "الرياض"},
    )
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "من هم أطباء القلب في فرع الرياض؟")
    assert response["specialty"] == "أمراض القلب"
    assert {d["name"] for d in response["doctors"]} == {"د. خالد المنصوري", "د. ليلى ناصر"}


def test_list_specializations_alexandria(repo):
    llm = FakeLLM(
        router={"intent": "action", "action": "list_specializations"},
        slots={"doctor_name": None, "specialty": None, "branch": "Alexandria"},
    )
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "What specializations are available at the Alexandria branch?")
    assert response["action"] == "list_specializations"
    assert response["branch"] == "Alexandria"
    assert set(response["specializations"]) == {
        "Neurology", "Cardiology", "Orthopedics", "Dermatology", "Pediatrics",
    }


def test_list_branches(repo):
    llm = FakeLLM(router={"intent": "action", "action": "list_branches"}, slots={})
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "What branches do you have?")
    assert response["action"] == "list_branches"
    assert {b["name"] for b in response["branches"]} == {"Cairo", "Alexandria", "Riyadh"}


def test_booking_unknown_doctor_offers_real_alternatives(repo):
    llm = FakeLLM(
        router={"intent": "action", "action": "book"},
        slots={"doctor_name": "Dr. Gregory House", "specialty": "Neurology", "branch": None},
    )
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "Book me with Dr. Gregory House in Neurology")
    assert "answer" in response  # clarification, never a fabricated booking
    assert "Dr. Sarah Hassan" in response["answer"]


def test_router_llm_failure_falls_back_to_heuristics(repo):
    class BrokenRouterLLM(FakeLLM):
        def chat(self, system, messages, json_mode=False):
            if "intent router" in system:
                raise RuntimeError("model server down")
            return super().chat(system, messages, json_mode)

    llm = BrokenRouterLLM(slots={"doctor_name": "Dr. Khalid", "specialty": None, "branch": None})
    graph, thread = run_graph(repo, llm)
    response = chat_turn(graph, thread, "I want to book an appointment with Dr. Khalid")
    assert response["action"] == "book_appointment"
    assert response["doctor_id"] == "DOC-003"
