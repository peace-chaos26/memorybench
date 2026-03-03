# ADR-002: Use gpt-4o-mini as the LLM Judge

**Status:** Accepted  
**Date:** 2024-01  
**Deciders:** Project author

---

## Context

The benchmark evaluates agent responses by comparing them to expected
answers. We need an automated judge that can handle paraphrased correct
answers (which exact-match would penalise unfairly).

## Options Considered

1. **Exact match** — no API cost, but penalises correct paraphrases
2. **ROUGE / BLEU** — overlap-based, no API cost, still misses semantic equivalence
3. **gpt-4o as judge** — highest quality, $5/1M tokens input
4. **gpt-4o-mini as judge** — lower quality, $0.15/1M tokens input (33× cheaper)
5. **Claude Haiku as judge** — similar to gpt-4o-mini in price/quality tier

## Decision

**gpt-4o-mini** as primary judge, with calibration runs to measure judge accuracy.

## Rationale

1. **Cost.** The full benchmark matrix (12 strategy combinations × 40 test
   cases × 6 probe questions = 2,880 judge calls). At gpt-4o pricing: ~$0.14.
   At gpt-4o-mini: ~$0.004. For development iteration (dozens of benchmark
   runs), this difference compounds.

2. **Binary judgement is a low-complexity task.** YES/NO correctness
   judgement does not require frontier-model reasoning. gpt-4o-mini handles
   binary classification from a well-designed rubric reliably.

3. **Judge calibration.** We validate judge accuracy on a 50-item held-out
   set with human labels (see `experiments/judge_calibration/`). If judge
   accuracy < 90%, we escalate to gpt-4o. Calibration showed 94% agreement
   for YES/NO correctness, validating the cheaper model choice.

4. **Consistency.** Using the same provider (OpenAI) for agent + judge
   reduces API surface area and authentication complexity.

## Consequences

- **Positive:** 33× cost reduction on judge calls without meaningful accuracy loss.
- **Negative:** Judge is not perfectly reliable. 6% error rate means ~170 of
  2,880 judgements may be wrong. This introduces noise in accuracy metrics —
  acceptable for relative comparisons between strategies.
- **Risk:** Judge bias toward outputs similar to OpenAI models. Since the
  agent also uses gpt-4o, there may be in-provider preference. To mitigate:
  judge prompt is rubric-based (fact-checking), not style-based.