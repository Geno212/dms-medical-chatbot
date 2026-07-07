# Multi-turn medical conversation ending in a booking (Arabic)

> **Note:** Action payloads below are exact system outputs. The conversational `answer`
> texts are representative of the prompted behavior; regenerate with
> `python scripts/generate_examples.py` against your live LLM endpoint.

**User:** عندي ألم في الصدر من الصباح وأحس بضيق في التنفس

**Chatbot:** *(the retrieved protocol is flagged `emergency` — triage escalation comes first, then the booking offer)*

```json
{
  "answer": "أتفهم قلقك، وسلامتك أهم شيء. ألم الصدر المصحوب بضيق في التنفس قد يكون علامة على حالة قلبية تستدعي الاهتمام الفوري — إذا كان الألم ضاغطاً أو يمتد إلى الذراع أو الفك أو يصاحبه تعرق أو دوخة، فتوجه فوراً إلى أقرب قسم طوارئ أو اتصل بالإسعاف. أما إذا كان الألم خفيفاً ومتقطعاً، فمن المهم تقييمه لدى طبيب القلب مع إجراء تخطيط للقلب في أقرب وقت. هل تودّ أن أحجز لك موعداً مع أحد أطباء القلب لدينا؟"
}
```

**User:** الألم مش مستمر، بييجي وبيروح

**Chatbot:**

```json
{
  "answer": "الحمد لله أنه غير مستمر، لكن الألم الصدري المتكرر حتى لو كان متقطعاً يُنصح بشدة بتقييمه لدى طبيب القلب مع إجراء تخطيط للقلب للاطمئنان. في هذه الأثناء تجنب المجهود الشديد، وإذا اشتد الألم أو صاحبه تعرق أو دوخة فلا تتردد في التوجه للطوارئ فوراً. هل أحجز لك موعداً مع أحد استشاريي أمراض القلب لدينا؟"
}
```

**User:** طيب احجزلي موعد مع دكتور قلب في القاهرة

**Chatbot:** *(exactly one cardiologist at the Cairo branch — verified booking payload returned directly)*

```json
{
  "action": "book_appointment",
  "doctor_id": "DOC-005",
  "doctor_name": "Dr. Ahmed Mostafa",
  "doctor_name_ar": "د. أحمد مصطفى",
  "specialty": "Cardiology",
  "specialty_id": "SP-CARD",
  "branch": "Cairo",
  "branch_id": "BR-CAI",
  "hospital": "Al-Mashreq Medical Group - Cairo Branch"
}
```
