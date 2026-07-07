# Hospital data lookups (doctors, specializations, branches)

> All payloads below are exact deterministic system outputs (no LLM in the
> response-construction path); entity mentions are resolved fuzzily and
> bilingually against the database before anything is returned.

**User:** Who are the cardiologists at the Riyadh branch?

**Chatbot:**

```json
{
  "action": "list_doctors",
  "specialty": "Cardiology",
  "branch": "Riyadh",
  "doctors": [
    { "name": "Dr. Khalid Al-Mansouri", "specialty": "Cardiology" },
    { "name": "Dr. Layla Nasser", "specialty": "Cardiology" }
  ]
}
```

**User:** What specializations are available at the Alexandria branch?

**Chatbot:** *(derived from which doctors actually practice at that branch)*

```json
{
  "action": "list_specializations",
  "branch": "Alexandria",
  "specializations": ["Cardiology", "Dermatology", "Neurology", "Orthopedics", "Pediatrics"]
}
```

**User:** ما هي فروع المستشفى؟

**Chatbot:** *(same action, Arabic query → Arabic field values)*

```json
{
  "action": "list_branches",
  "hospital": "Al-Mashreq Medical Group",
  "branches": [
    { "name": "الإسكندرية", "address": "45 طريق الجيش، سيدي جابر، الإسكندرية، مصر", "phone": "+20-3-5450-2000" },
    { "name": "القاهرة", "address": "12 كورنيش النيل، المعادي، القاهرة، مصر", "phone": "+20-2-2525-1000" },
    { "name": "الرياض", "address": "طريق الملك فهد، حي العليا، الرياض، المملكة العربية السعودية", "phone": "+966-11-460-3000" }
  ]
}
```
