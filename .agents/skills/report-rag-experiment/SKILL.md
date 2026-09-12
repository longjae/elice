---
name: report-rag-experiment
description: Analyze saved Part C baseline and improved RAG evaluation runs and write the Hypothesis, Result, Analysis, and Next Steps report. Use only after a baseline result exists.
---

# Report the RAG experiment

1. Refuse to compare runs that use different Gold Sets or incompatible metric definitions.
2. Load saved baseline and improved reports; do not rerun them implicitly.
3. Verify model IDs, configs, question counts, failures, and timestamps.
4. Present absolute scores and deltas for every primary metric.
5. Break results down by question type and inspect meaningful regressions.
6. Separate observed evidence from interpretation.
7. Write README sections in this order: Hypothesis, Result, Analysis, Next Steps.
8. Report flat or negative results honestly; improvement is not a completion criterion.

Do not select only favorable questions, overwrite raw reports, or change the Gold Set
after seeing improved-run results without documenting a new experiment version.
