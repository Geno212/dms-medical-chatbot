# Voice input: spoken Arabic → transcription → regular chat processing

> Flow: the user records with the Chainlit microphone button (or attaches an
> audio file / uses `/voice <path>` in the CLI). Audio is transcribed **locally**
> by faster-whisper, echoed back as a transcription payload, and then processed
> as a normal chat message through the same graph.

**Request:** 🎤 audio recording (spoken Arabic)

**Step 1 — transcription payload:**

```json
{
  "transcribed_text": "عندي ألم في الصدر من الصباح وأحس بضيق في التنفس"
}
```

**Step 2 — the transcribed text is sent to the chat as a regular message, and the
medical RAG path answers it** (this symptom set is flagged `emergency` by the
knowledge base, so the answer leads with urgent-care guidance):

```json
{
  "answer": "أتفهم قلقك، وسلامتك أهم شيء. ألم الصدر المصحوب بضيق في التنفس قد يكون علامة على حالة قلبية تستدعي الاهتمام الفوري — إذا كان الألم شديداً أو يمتد إلى الذراع أو الفك أو يصاحبه تعرق، فتوجه فوراً إلى أقرب قسم طوارئ. وإذا كانت الأعراض أخف من ذلك، فمن المهم تقييمها لدى طبيب القلب في أقرب وقت. هل تودّ أن أحجز لك موعداً مع أحد أطباء القلب لدينا؟"
}
```

**Notes**

* Whisper auto-detects the spoken language (Arabic/English, including
  code-switching) — no language selection needed.
* `WHISPER_MODEL=small` is the CPU-friendly default; `large-v3` gives the best
  Arabic accuracy if you have a GPU.
* No audio ever leaves the machine.
