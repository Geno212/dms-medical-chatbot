"""Knowledge-base retrieval (lexical channel — runs without an embedding server)."""

from app.llm import extract_json
from app.vectorstore import ProtocolRetriever


def make_retriever(repo):
    return ProtocolRetriever(repo, embedder=None)


def test_english_headache_fever_hits_neurology_protocol(repo):
    results = make_retriever(repo).search("I've been having a severe headache and fever for two days")
    assert results and results[0]["id"] == "PROT-001"
    assert results[0]["specialization_id"] == "SP-NEUR"


def test_arabic_chest_pain_hits_cardiology_emergency(repo):
    results = make_retriever(repo).search("عندي ألم في الصدر من الصباح وأحس بضيق في التنفس")
    assert results and results[0]["id"] == "PROT-002"
    assert results[0]["triage"] == "emergency"


def test_arabic_child_fever_hits_pediatrics(repo):
    results = make_retriever(repo).search("ابني عنده سخونية عالية من امبارح")
    assert results and results[0]["specialization_id"] == "SP-PEDI"


def test_irrelevant_query_returns_weak_or_no_results(repo):
    results = make_retriever(repo).search("what is the capital of France")
    assert all(r["score"] < 0.3 for r in results)


# --- JSON extraction robustness (small local models are messy) ---

def test_extract_json_plain():
    assert extract_json('{"intent": "medical", "action": "none"}') == {"intent": "medical", "action": "none"}


def test_extract_json_code_fence():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_prose():
    assert extract_json('Sure! Here is the JSON:\n{"a": {"b": 2}}\nHope that helps.') == {"a": {"b": 2}}


def test_extract_json_garbage_returns_none():
    assert extract_json("I cannot do that.") is None
    assert extract_json("") is None
