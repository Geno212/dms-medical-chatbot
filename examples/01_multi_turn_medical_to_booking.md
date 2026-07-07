# Multi-turn medical conversation ending in a booking (English)

> **Note:** Action payloads below are exact system outputs (asserted in `tests/test_graph.py`).
> The conversational `answer` texts are representative of the prompted behavior; run
> `python scripts/generate_examples.py` with your LLM endpoint up to regenerate this file
> with live model output.

**User:** I've been having a severe headache and fever for two days, what should I do?

**Chatbot:**

```json
{
  "answer": "I'm sorry to hear you've been dealing with this for two days — that sounds exhausting. A severe headache accompanied by fever lasting more than 24 hours may indicate an inflammation or infection that needs a medical examination. For now, rest fully, drink plenty of water, and avoid bright light. If the headache worsens or you notice neck stiffness, confusion, or vomiting, please seek care promptly. Would you like me to book you an appointment with one of our Neurology doctors?"
}
```

**User:** It gets worse when I look at bright light, is that serious?

**Chatbot:**

```json
{
  "answer": "Light sensitivity together with a severe headache and fever is a sign worth taking seriously — it can accompany migraine, but with fever it should be checked to rule out an infection. Please continue resting in a dim room and staying hydrated, and don't delay an evaluation if it persists. Shall I book you an appointment with one of our Neurology specialists so a doctor can examine you properly?"
}
```

**User:** Yes, please book me an appointment

**Chatbot:** *(no doctor was ever named — the system offers only real, verified options from the database)*

```json
{
  "answer": "We have more than one suitable option: Dr. Sarah Hassan (Neurology - Cairo branch); Dr. Omar El-Sayed (Neurology - Alexandria branch). Who would you like to book with?"
}
```

**User:** Dr. Sarah please, the Cairo branch works for me

**Chatbot:** *(verified structured booking payload — note the specialty was carried from the medical discussion, never typed by the user)*

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
