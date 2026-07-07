# AI-Powered Medical Chatbot — Al-Mashreq Medical Group

A context-aware, bilingual (Arabic/English) medical chatbot for a hospital group.
Patients describe symptoms in natural language and receive empathetic,
**knowledge-base-grounded** recommendations; when they want to *act* — book an
appointment, find a doctor, list branches — the bot returns **verified structured
JSON** built from the hospital's real data. Voice input is transcribed locally.

**Privacy-first by design: the LLM, the speech-to-text, and the database all run
locally. No patient utterance ever leaves the machine.**

---

## 1. Architecture

```
                        ┌──────────────────────┐
  Chainlit UI ──text──► │      LangGraph       │
  (or CLI)    ──audio─► │  ┌────────────────┐  │
       │                │  │     router      │  │  intent + language
  faster-whisper        │  └───┬────┬────┬──┘  │
  (local STT)           │      ▼    ▼    ▼     │
                        │ medical action other │
                        │  (RAG) (verified)    │
                        └───┬──────┬───────────┘
                            ▼      ▼
                  hybrid retrieval  entity resolution
                  (vector+lexical)  (bilingual fuzzy match)
                            └──────┬┘
                                   ▼
                          SQLite  (branches, specializations,
                                   doctors, medical protocols + embeddings)
```

| Component | Choice | Why |
|---|---|---|
| Orchestration | **LangGraph** | Explicit stateful graph: routing, per-thread conversation memory (checkpointer), and a clean place for the clinical-context slot-filling that makes multi-turn booking work. |
| LLM | **Ollama** (`qwen2.5:7b-instruct`, OpenAI-compatible API) | Qwen2.5 has strong Arabic among open models and reliable JSON output. Local inference = medical privacy. The client is endpoint-agnostic: point `LLM_BASE_URL` at LM Studio, Groq, or OpenAI and nothing else changes. |
| Knowledge base + data | **SQLite** (default) or **Supabase/PostgreSQL + pgvector** | Two interchangeable backends behind one repository interface. SQLite gives reviewers zero-setup clone-and-run; setting `DATABASE_URL` switches the whole system to Supabase for a cloud deployment — see §7. |
| Embeddings | **bge-m3** (via Ollama) | Genuinely multilingual — one vector space for Arabic and English symptom text. |
| Retrieval | **Hybrid: dense cosine + curated bilingual symptom keywords** | Keywords keep short Arabic symptom phrases precise and keep the bot grounded even if the embedding model is missing; dense vectors catch paraphrases. |
| STT | **faster-whisper** (local) | Whisper is the reference open model for Arabic speech; auto language detection; runs on CPU with the `small` model. |
| UI | **Chainlit** | Native microphone streaming + file upload, chat UI out of the box. |

### The core design decision: LLM understands, code decides

The LLM is **never** trusted to produce a final action payload. It does three
jobs only: classify intent, extract *verbatim* slot mentions ("Dr. Sarah",
"قلب", "alex"), and phrase grounded medical answers. Deterministic code then:

1. **Resolves** mentions against the database with bilingual fuzzy matching
   (Arabic orthography normalization, title stripping, partial-name and alias
   matching) — so "د. سارة" → `Dr. Sarah Hassan (DOC-001)`.
2. **Refuses to guess**: an ambiguous mention ("Dr. Hassan" matches two
   doctors) or an unknown doctor produces a clarification question listing
   *real* options — never a fabricated booking.
3. **Builds the JSON** exclusively from database rows, with IDs
   (`doctor_id`, `branch_id`, `specialty_id`) a downstream booking system needs.

This makes hallucinated bookings structurally impossible, not just unlikely.

### Multi-turn context & slot-filling

The graph state is checkpointed per conversation thread. Besides the message
history, a small **clinical context** survives across turns: which specialty the
knowledge base pointed to for the patient's symptoms. So this works with the
user never repeating anything:

> *"I've had a severe headache and fever for two days"* → grounded advice + Neurology suggested
> *"yes, book me an appointment"* → the bot offers the actual neurologists on staff
> *"Dr. Sarah please"* → verified `book_appointment` payload for `DOC-001`

### Safety / triage layer

Every protocol in the knowledge base carries a triage level. If retrieval hits
an `emergency` protocol (e.g. chest pain + shortness of breath), the answer is
prompted to lead with urgent-care/ambulance guidance before any booking talk.
Medical answers are constrained to the retrieved hospital protocols and always
recommend specialist evaluation instead of diagnosing.

## 2. Response contract

Every reply is one of two disjoint shapes (per the task spec):

```jsonc
{ "answer": "conversational medical text, in the user's language" }
// or
{ "action": "book_appointment" | "list_doctors" | "list_specializations" | "list_branches", ... }
```

Booking example (all values verified against the DB):

```json
{
  "action": "book_appointment",
  "doctor_id": "DOC-001",
  "doctor_name": "Dr. Sarah Hassan",
  "doctor_name_ar": "د. سارة حسن",
  "specialty": "Neurology",
  "specialty_id": "SP-NEUR",
  "branch": "Cairo",
  "branch_id": "BR-CAI",
  "hospital": "Al-Mashreq Medical Group - Cairo Branch"
}
```

Voice input produces `{ "transcribed_text": "..." }`, then the text is processed
as a regular message. See [examples/](examples/) for full transcripts.

## 3. Prompt design strategy

Four prompts, each with a single narrow job (see `app/graph/prompts.py`):

* **Router** — few-shot intent classification to strict JSON, with the explicit
  rule that an affirmative reply to a booking offer ("نعم", "yes") is a `book`
  action. Falls back to bilingual keyword heuristics if the LLM call fails.
* **Slot extractor** — extracts doctor/specialty/branch *verbatim, without
  translating*, `null` when absent, "never invent values". Verification is
  code's job, which is what makes small local models safe to use here.
* **Medical answer** — persona ("informed member of the hospital staff"),
  retrieved protocols injected as *the only permitted medical source*, hard
  rules: match the user's language, lead with empathy, no definitive diagnosis,
  end with a booking offer phrased as a question. A triage note is injected
  when retrieval flags an emergency.
* **Smalltalk/identity** — hospital-aware persona for greetings and meta questions.

JSON robustness: `json_mode` is requested from the endpoint *and* replies pass
through a tolerant extractor (code fences, surrounding prose, balanced-brace
scan) because small local models are messy.

## 4. Setup & run

```bash
# 1. Install
pip install -r requirements.txt

# 2. Get a local LLM + embedding model (default configuration)
ollama pull qwen2.5:7b-instruct
ollama pull bge-m3

# 3. Configure (optional — defaults work with local Ollama)
cp .env.example .env

# 4. Seed the database (computes protocol embeddings via the endpoint)
python scripts/seed_db.py

# 5a. Web UI (mic button enabled)
chainlit run app/chainlit_app.py -w

# 5b. or terminal chat
python -m app.cli
```

No Ollama? Point `.env` at any OpenAI-compatible endpoint (LM Studio, Groq,
OpenAI) — see `.env.example`. No embedding model? `seed_db.py --skip-embeddings`
still works: retrieval falls back to the bilingual lexical channel.

> **Low-VRAM GPU tip:** on machines with a small discrete GPU (≤2 GB VRAM),
> Ollama may crash trying to partially offload the 7B model. Force stable
> CPU-only inference by starting the server with the GPU hidden:
> `CUDA_VISIBLE_DEVICES=-1 ollama serve` (PowerShell:
> `$env:CUDA_VISIBLE_DEVICES="-1"; ollama serve`).

**Voice**: `pip install faster-whisper` (already in requirements). First use
downloads the Whisper model. In the web UI use the mic button or attach an
audio file; in the CLI use `/voice path/to/audio.wav`.

## 5. Tests

```bash
python -m pytest tests/ -q     # 33 tests
```

The suite injects a scripted fake LLM, so the full pipeline — routing,
retrieval grounding, bilingual entity resolution, multi-turn slot-filling,
verified payload construction, ambiguity refusal, heuristic fallback — is
tested deterministically without a model server.

## 5b. Pipeline logging (observability)

Every graph node logs its decision to a dedicated `chatbot` logger
(`app/logging_setup.py`), so a single conversation turn prints a visible
trace of the "LLM understands, code decides" split as it happens:

```
chatbot INFO: router: message="I've been having a severe headache and fever..." language=en intent=medical action=none source=llm
chatbot INFO: retrieval: query="I've been having a severe headache..." hits=[('PROT-001', 0.8013, 'urgent'), ('PROT-013', 0.5174, 'routine'), ...]
chatbot INFO: clinical context updated: suggested_specialization=Neurology
chatbot INFO: router: message='Yes, please book me an appointment' language=en intent=action action=book source=llm
chatbot INFO: slot extraction: {'doctor_name': None, 'specialty': 'Neurology', 'branch': None}
chatbot INFO: entity resolution: branch=None -> None (0 candidates) | specialty='Neurology' -> Neurology (1 candidates)
chatbot INFO: 2 doctors match specialty Neurology -> asking user to choose
chatbot INFO: slot extraction: {'doctor_name': 'Dr. Sarah Hassan', 'specialty': 'Neurology', 'branch': 'Cairo'}
chatbot INFO: doctor resolved: 'Dr. Sarah Hassan' -> Dr. Sarah Hassan (DOC-001)
```
(real trace from `scripts/generate_examples.py` — the transcript in
`examples/01_multi_turn_medical_to_booking.md`)

Set `LOG_LEVEL=DEBUG` in `.env` to also see raw LLM JSON output, or
`LOG_LEVEL=WARNING` to silence it (only triage escalations and fallbacks
still print). Runs automatically for the CLI, Chainlit UI, and any script
that calls `build_graph()`.

## 6. Test dataset

`data/hospital_dataset.json` — AI-generated, fully bilingual: **Al-Mashreq
Medical Group** (مجموعة المشرق الطبية), 3 branches (Cairo, Alexandria, Riyadh),
8 specializations, 18 doctors, and 14 symptom→specialty medical protocols with
triage levels and curated Arabic/English symptom keywords. Edit it and re-run
`scripts/seed_db.py` to change the hospital.

## 7. Running on Supabase (PostgreSQL + pgvector)

The Postgres backend is already implemented (`app/db_postgres.py`) behind the
same repository interface — no code changes needed:

```bash
# 1. Create a free project at supabase.com
# 2. Dashboard -> Connect -> Connection String -> copy the Session pooler URI
# 3. In .env:
DATABASE_URL=postgresql://postgres.xxxx:PASSWORD@aws-0-region.pooler.supabase.com:5432/postgres
# 4. Seed the cloud database (creates schema + pgvector extension automatically)
python scripts/seed_db.py
```

Everything — graph, matching, retrieval, UI — now runs against Supabase.
Protocol embeddings are stored in a pgvector `vector` column; at the current
KB size dense ranking happens in-process, and at scale it moves into SQL
(`ORDER BY embedding <=> $1`) without touching the interface.

## 8. Production path
* **MemorySaver → Postgres checkpointer** for conversation persistence across
  restarts.
* Real booking write-path (appointments table + availability), auth, and an
  evaluation harness (golden conversations asserting routing + payload
  correctness) are the natural next steps.

## Project layout

```
app/
  config.py          env-driven configuration
  db.py              SQLite repository (hospital data)
  matching.py        bilingual fuzzy entity resolution
  vectorstore.py     hybrid retrieval (dense + lexical)
  llm.py             OpenAI-compatible client + robust JSON extraction
  stt.py             faster-whisper transcription (lazy, optional)
  graph/             LangGraph: state, prompts, nodes, wiring
  cli.py             terminal chat
  chainlit_app.py    web UI with mic + audio upload
data/                dataset JSON + seeded SQLite DB
scripts/             seed_db.py, generate_examples.py
tests/               31 deterministic tests (fake LLM)
examples/            deliverable conversation transcripts
```
