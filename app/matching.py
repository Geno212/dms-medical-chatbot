"""Bilingual fuzzy entity resolution.

LLMs extract what the user *said* ("Dr. Sarah", "قلب", "alex"); this module
resolves it against what actually exists in the hospital database
("Dr. Sarah Hassan", Cardiology, Alexandria). Verification lives in
deterministic code, not in the LLM, so the bot can never book a doctor that
does not exist.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

# Arabic normalization tables
_ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")  # tashkeel + tatweel
_ALEF_VARIANTS = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})

# Applied AFTER normalize(), which maps ة->ه — so title variants use ه.
_TITLE_PATTERNS = re.compile(
    r"^(dr\.?|doctor|prof\.?|professor|د\.?|دكتور(ه)?|الدكتور(ه)?)\s+",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Case-fold, strip punctuation, and normalize Arabic orthography."""
    text = unicodedata.normalize("NFKC", text or "").lower().strip()
    text = _ARABIC_DIACRITICS.sub("", text)
    text = text.translate(_ALEF_VARIANTS)
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = re.sub(r"[^\w\s؀-ۿ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_title(name: str) -> str:
    normalized = normalize(name)
    while True:
        stripped = _TITLE_PATTERNS.sub("", normalized)
        if stripped == normalized:
            return stripped
        normalized = stripped


def _token_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if len(a) >= 3 and (a.startswith(b) or b.startswith(a)):
        return 0.9
    return SequenceMatcher(None, a, b).ratio()


def name_score(query: str, candidate: str) -> float:
    """How well a (possibly partial) query matches a candidate name.

    Every query token must find a good counterpart among candidate tokens;
    a partial name like "sarah" scores highly against "sarah hassan".
    """
    q_tokens = strip_title(query).split()
    c_tokens = strip_title(candidate).split()
    if not q_tokens or not c_tokens:
        return 0.0
    scores = [max(_token_sim(q, c) for c in c_tokens) for q in q_tokens]
    if min(scores) < 0.55:  # one query token found no counterpart at all
        return 0.0
    return sum(scores) / len(scores)


def _entity_score(query: str, entity: dict[str, Any], name_keys: tuple[str, ...] = ("name_en", "name_ar")) -> float:
    candidates = [entity[k] for k in name_keys if entity.get(k)]
    candidates += entity.get("aliases", [])
    return max((name_score(query, c) for c in candidates), default=0.0)


def resolve_entity(
    query: str | None,
    entities: list[dict[str, Any]],
    threshold: float = 0.75,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Resolve a user-mentioned name against DB entities.

    Returns (match, candidates): a single confident match, or the list of
    plausible candidates when the mention is ambiguous (caller should ask
    the user to clarify rather than guess).
    """
    if not query or not query.strip():
        return None, []
    scored = [(entity, _entity_score(query, entity)) for entity in entities]
    scored = [(e, s) for e, s in scored if s >= threshold]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    if not scored:
        return None, []
    top_entity, top_score = scored[0]
    # Ambiguous if a different entity scores nearly as high as the winner.
    runners_up = [e for e, s in scored[1:] if s >= top_score - 0.05]
    if runners_up:
        return None, [top_entity, *runners_up]
    return top_entity, [top_entity]


def resolve_doctor(
    query: str | None,
    doctors: list[dict[str, Any]],
    specialization_id: str | None = None,
    branch_id: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Resolve a doctor mention, using specialty/branch as disambiguators.

    Filters are relaxed in order of trust when they yield no match: the
    branch/specialty the *user stated this turn* is trusted more than a
    specialty carried over from earlier context. Concretely we try, in order:
      1. specialty + branch   (both constraints)
      2. branch only          (a carried-over specialty may be wrong — e.g. the
                               patient described symptoms pointing elsewhere but
                               then named a doctor in a specific branch)
      3. specialty only
      4. no filter            (named a doctor while misremembering both)
    The first filter set that produces a match or candidates wins, so an
    explicit branch is never silently discarded to force a name-only match.
    """
    def _pool(spec: str | None, branch: str | None) -> list[dict[str, Any]]:
        return [
            d for d in doctors
            if (not spec or d["specialization_id"] == spec)
            and (not branch or d["branch_id"] == branch)
        ]

    filter_sets = [(specialization_id, branch_id)]
    if specialization_id and branch_id:
        filter_sets.append((None, branch_id))   # keep the stated branch first
        filter_sets.append((specialization_id, None))
    if specialization_id or branch_id:
        filter_sets.append((None, None))

    seen: set[tuple[str | None, str | None]] = set()
    for spec, branch in filter_sets:
        if (spec, branch) in seen:
            continue
        seen.add((spec, branch))
        match, candidates = resolve_entity(query, _pool(spec, branch))
        if match or candidates:
            return match, candidates
    return None, []
