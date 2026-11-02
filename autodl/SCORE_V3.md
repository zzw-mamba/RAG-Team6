# score-v3: local evidence and bounded reasoning ablation

This candidate is unvalidated on the target GPU/Kaggle. It does not promise 0.445 or any gain.

Code inspection found that the old JSON decoder forces answer before explanation. This prevents the generated explanation from informing the already generated answer. Coverage also appends global corpus hits to the question-local passages. These are hypotheses to test, not confirmed causes of the Kaggle score.

The new suite compares legacy baseline, incumbent (the previous planned policy), local_direct, local_reasoned and local_knowledge. Local policies retain every supplied passage before token budgeting and do not use global search or a query planner. local_reasoned changes only output order relative to local_direct. local_knowledge additionally permits explicitly identified prior knowledge. Explanation is limited to 320 characters by both the prompt and schema, preventing the previous unbounded genealogy failure. No labels enter inference.

`bash run_score_v3.sh check /root/autodl-fs/zzw/configs/new-server.json` creates `/root/autodl-fs/zzw/configs/score-v3.json` using the existing prepared data sizes/seed/work directory. It compares five policies on existing tune, freezes the winner, then compares it with both baseline and incumbent on confirmation. It does not run the test set. Results live under `work_dir/optimization/score-v3/`. One Qwen model is loaded per stage, no critic or training is required.

IMPORTANT: existing confirmation has already been inspected and is no longer an untouched final test. Reusing it is a development regression check, not evidence of generalization. Do not tune repeatedly against it or Kaggle public scores. A larger truly untouched holdout is needed before confident claims. Old v2 local EM is not the Kaggle scorer and the tune distribution may differ from test. Obtain the old selection/confirmation and teammate's pipeline to diagnose that gap.

After reviewing results, `bash run_score_v3.sh predict /root/autodl-fs/zzw/configs/new-server.json` first runs the existing real sequential/batch pilot, then only starts full prediction if quality and speed checks pass. Default batch=8; set BATCH_SIZE=16 MAX_BATCH_TOKENS=294912 to test batch16. It writes score-v3/<selected>/test.csv, preserving v2 predictions.

Changing package source invalidates old source signatures. Do not update an actively running v2 checkout or try to resume v2 with this code. Finish old jobs first or keep the old checkout unchanged and run v3 in a separate checkout. No weights, labels or previous predictions need to be deleted. No new Python packages are needed.

Before a long test run inspect tune and confirmation metrics, especially entity questions, context coverage and unknown rate. A tiny tie in EM is not proof of an improvement. GPU sharing still affects runtime. Synthetic tests verify ordering/isolation, not accuracy.

Offline evidence audit (no GPU):

```bash
python scripts/audit_score_evidence.py --config /root/autodl-fs/zzw/configs/new-server.json --traces /root/autodl-fs/zzw/rag-team6-work/optimization/score-v2/planned/confirmation.jsonl --output /root/autodl-fs/zzw/rag-team6-work/optimization/score-v3-evidence-audit.json
```

Compare gold_in_full_local vs gold_in_visible. local_gold_lost suggests retrieval/budgeting loss; low gold_in_full_local means provided-only candidates may lack answer evidence and global retrieval needs improvement. wrong_despite_visible_gold flags answer reasoning errors, but does not prove all required hops are present. The report contains labels and is strictly offline, never an inference input. Existing v2 artifacts can be audited read-only despite new source signatures.
