"""Bilingual fuzzy entity resolution against the seeded hospital data."""

from app.matching import name_score, normalize, resolve_doctor, resolve_entity, strip_title


def test_normalize_arabic_variants():
    assert normalize("الإسكندريّة") == normalize("الاسكندريه")
    assert normalize("أحمد") == normalize("احمد")


def test_strip_titles():
    assert strip_title("Dr. Sarah Hassan") == "sarah hassan"
    assert strip_title("د. سارة حسن") == normalize("سارة حسن")
    assert strip_title("الدكتورة ليلى ناصر") == normalize("ليلى ناصر")


def test_partial_name_scores_high():
    assert name_score("Dr. Sarah", "Dr. Sarah Hassan") > 0.9
    assert name_score("sara hasan", "Dr. Sarah Hassan") > 0.75  # typo tolerance
    assert name_score("Dr. John", "Dr. Sarah Hassan") == 0.0


def test_resolve_doctor_partial_english_unique_when_specialty_given(repo):
    # Two "Dr. Sarah Hassan" exist (Neurology + Dermatology). The name alone is
    # ambiguous; the specialty filter pins it to the neurologist DOC-001.
    doctor, _ = resolve_doctor("Dr. Sarah", repo.list_doctors(), specialization_id="SP-NEUR")
    assert doctor and doctor["id"] == "DOC-001"


def test_resolve_doctor_arabic_unique_when_specialty_given(repo):
    doctor, _ = resolve_doctor("د. سارة", repo.list_doctors(), specialization_id="SP-NEUR")
    assert doctor and doctor["id"] == "DOC-001"


def test_resolve_same_name_different_specialty_is_ambiguous(repo):
    # Edge case: "Dr. Sarah" matches two real people in different specialties.
    # The resolver must NOT guess — it returns both as candidates.
    doctor, candidates = resolve_doctor("Dr. Sarah", repo.list_doctors())
    assert doctor is None
    ids = {c["id"] for c in candidates}
    assert {"DOC-001", "DOC-021"} <= ids  # Neurology + Dermatology Sarahs


def test_resolve_same_name_pinned_by_specialty(repo):
    # The same "Dr. Mona Adel" name exists in Cardiology (DOC-006) and
    # Pediatrics (DOC-020); a specialty disambiguates to exactly one.
    cardio, _ = resolve_doctor("Dr. Mona", repo.list_doctors(), specialization_id="SP-CARD")
    assert cardio and cardio["id"] == "DOC-006"
    pedia, _ = resolve_doctor("Dr. Mona", repo.list_doctors(), specialization_id="SP-PEDI")
    assert pedia and pedia["id"] == "DOC-020"


def test_same_name_same_specialty_needs_branch(repo):
    # Edge case: two "Dr. Layla Nasser" both in Cardiology, different branches.
    # Name + specialty is still ambiguous — only the branch disambiguates.
    docs = repo.list_doctors()
    match, candidates = resolve_doctor("Dr. Layla", docs, specialization_id="SP-CARD")
    assert match is None
    branches = {c["branch_id"] for c in candidates}
    assert {"BR-CAI", "BR-RUH"} <= branches  # the clarification spans both branches


def test_same_name_same_specialty_pinned_by_branch(repo):
    docs = repo.list_doctors()
    cairo, _ = resolve_doctor("Dr. Layla", docs, specialization_id="SP-CARD", branch_id="BR-CAI")
    riyadh, _ = resolve_doctor("Dr. Layla", docs, specialization_id="SP-CARD", branch_id="BR-RUH")
    assert cairo and cairo["id"] == "DOC-022"
    assert riyadh and riyadh["id"] == "DOC-004"
    assert cairo["id"] != riyadh["id"]


def test_resolve_doctor_ambiguous_surname(repo):
    # "Hassan" appears in Dr. Sarah Hassan and Dr. Hassan El-Guindy:
    # the resolver must ask, not guess.
    doctor, candidates = resolve_doctor("Dr. Hassan", repo.list_doctors())
    assert doctor is None
    assert len(candidates) >= 2


def test_resolve_doctor_disambiguated_by_specialty(repo):
    neurology_id = "SP-NEUR"
    doctor, _ = resolve_doctor("Dr. Hassan", repo.list_doctors(), specialization_id=neurology_id)
    assert doctor and doctor["id"] == "DOC-001"


def test_resolve_specialty_by_arabic_alias(repo):
    spec, _ = resolve_entity("قلب", repo.list_specializations())
    assert spec and spec["id"] == "SP-CARD"


def test_resolve_specialty_by_english_alias(repo):
    spec, _ = resolve_entity("cardiologists", repo.list_specializations())
    assert spec and spec["id"] == "SP-CARD"


def test_resolve_branch_nickname(repo):
    branch, _ = resolve_entity("alex", repo.list_branches())
    assert branch and branch["id"] == "BR-ALX"


def test_resolve_branch_arabic(repo):
    branch, _ = resolve_entity("الاسكندريه", repo.list_branches())
    assert branch and branch["id"] == "BR-ALX"


def test_unknown_doctor_returns_nothing(repo):
    doctor, candidates = resolve_doctor("Dr. Gregory House", repo.list_doctors())
    assert doctor is None and candidates == []
