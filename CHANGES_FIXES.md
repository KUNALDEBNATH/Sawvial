# SAWVIAL fixes (backend answer quality)

All in `api.py` and `llm/rag_chat.py`:

1. **Relevance cut-off too high** - `MIN_RETRIEVAL_SCORE` 0.11 -> 0.03. TF-IDF scores for good
   matches are 0.05-0.25, so ordinary questions ("how much water does my field need this week")
   were rejected with "couldn't find relevant, reliable evidence".
2. **Follow-up detection was too greedy** - words like "this", "that", "it" anywhere in a question
   made it a "follow-up", mixing the previous topic into retrieval. Pronouns now only count in
   short messages (<= 7 words) or after openers like "and", "what about", "why".
3. **Out-of-scope filter missed farming words** - rice, paddy, wheat, maize, tomato, seed, sow,
   harvest, weeds, pesticide, etc. are now recognised.
4. **Answers thrown away for small LLM slips** - invented citation ids, `[A_1, B_2]` style
   citations, empty output, or not repeating the crop name used to return the dead-end message.
   Now invented ids are ignored and, if the LLM output is unusable (or it says "not enough
   information" although retrieval was strong), the best matching evidence sentences are returned
   with their source id instead.
5. **`repetition_penalty` 1.1 -> 1.0** in `llm/rag_chat.py` - the penalty mangles repeated tokens
   such as `SWM_000123`, which caused the invented-citation rejections.
6. **Translator errors no longer crash the request** - falls back to the English answer.
7. **Misleading numbers** - dataset "findings" like "a range of 2.0-3.0 was reported" are auto-generated
   and unit-less; they are no longer shown to the LLM (`llm/rag_chat.py`).
8. **Answer style** - system prompt (`llm/common.py`) now asks for 2-4 sentences, no dangling
   "found that..." sentences, no author names, no unit-less numbers.
9. **Raw `[SWM_000671]` in canned answers** - citation tags are stripped from the potato-irrigation and
   crop-list answers (sources are still listed under "Sources").
