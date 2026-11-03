# score-v5: public-question routing candidate

v4 confirmation observed +8 correct entity questions but -10 correct boolean questions; unknowns increased from9 to76. This does not isolate whether prompt order, evidence selection or another interaction caused rejection. Do not run v4 full test after its failed check.

v5 changes:

- `baseline`: prior v2 planned, identical semantic config to v4 baseline.
- `hybrid_direct`: v4 hybrid retrieval with the old answer-first knowledge prompt. This tests whether retrieval can help without the changed explanation-first prompt.
- `routed_reasoned`: explicit question-text heuristic: initial Is/Are/Was/Were/Do/Does/Did/Has/Have/Had/Can/Could/Will/Would/Should => old planned; otherwise v4 hybrid reasoning. The rule does not consult labels, IDs, predictions or correctness. Which/Who questions asking for an option remain entity route.
- Optional `routed_bge` if BGE_PATH is explicitly set, with BGE disabled for the boolean route.

No DBpedia, new fact sources or retraining. No generic forced yes/no or replacement of unknown with guesses. An auxiliary-first question can still request a named option; this is a heuristic, not a perfect type classifier. Do not claim the expected aggregate 65+33 as a measured new model result. Actual routing must be run and scored.

Per-controller cfg/retriever state is restored even after exceptions. Shared Qwen cfg is never mutated by routing; all candidates use the same16384 input budget. Output token budgets and schemas are passed per request. Model caches include route identity. Critic mode and externally supplied initials are intentionally refused for routed candidates.

## Usage

Use a new checkout and keep old results. In autodl/:

```bash
nohup bash run_score_v5.sh check /root/autodl-fs/zzw/configs/new-server.json \
  > /root/autodl-fs/zzw/logs/score-v5.log 2>&1 < /dev/null &
```

Existing old venv/data/index/weights are reused. Default no BGE or package downloads. To enable existing BGE weights, set BGE_PATH/BGE_DEVICE before the first run and preserve them for all stages. New config: /root/autodl-fs/zzw/configs/score-v5.json. Outputs: work_dir/optimization/score-v5/. Only tune+confirmation run by default. Predict is a separate stage and requires quality and fresh batch-pilot checks.

Tune freezes one winner before confirmation; confirmation requires overall and entity EM not below planned. These existing splits have been repeatedly inspected and are now development checks, not independent generalization evidence. Before strong claims, evaluate a fresh disjoint holdout excluded from the corpus or use final unlabelled test under frozen rules. Never rebuild the corpus with heldout labels or select route per answer correctness. Actual GPU quality, speed and Kaggle score remain unvalidated locally.
