# score-v4: dataset-only hybrid retrieval

No DBpedia, online encyclopedia, external answer lookup or label-based retrieval. Only the existing train-context FTS corpus and each question's public contexts are searched. A pretrained BGE reranker is an optional additional model, not an added facts database. No promised Kaggle score or target GPU accuracy.

## Changes

- `title_search` searches the FTS title field with bound quoted AND tokens, preserving year and diacritics; existing corpus.sqlite needs no rebuild.
- Queries include matching supplied titles, quoted names and bounded explicit name spans. The Qwen planner still supplies targeted missing-hop queries. No answer labels enter queries.
- Each original/planner query maintains its own candidate ranking. Reciprocal rank fusion combines lexical (1.0), title (1.5), entity-title (1.5) and optional BGE (2.0) channels with denominator 20+rank.
- Four ranked question-local passages get reserved places, then up to five original-query hits and three per planned branch are interleaved, then filled to 20. At most two passages of a title; actual tokenizer budgeting still applies. This can lose useful local evidence, so measure it rather than assume safety.
- BGE reranks up to 48 fused candidates independently for each relation, including local candidates (not just global hits). It uses existing Transformers sequence-classification integration, max length512 and batch4. Pair truncation remains a limitation.
- Responses use the bounded explanation-first JSON from v3 (max320 characters), then a short answer and exact document citations. Prior model knowledge remains enabled as in the old planned incumbent.
- No BGE_PATH: compare two policies, old planned and hybrid_reasoned. With BGE_PATH: also compare hybrid_bge. No silent BGE fallback or automatic downloads.
- Current planned is called `baseline` in v4, not the weak legacy baseline. Tune freezes a winner; confirmation checks BOTH overall and entity EM against planned. A tie is no proof of improvement. Confirmation has already been inspected and is only a development regression check, not an untouched statistical test. Heldout batches differ from Kaggle.

## Run

Use a new checkout; changing package source invalidates v2/v3 signatures. Keep old checkout/venv/results. Do not run competing GPU jobs for this project while benchmarking.

```bash
cd /root/autodl-fs/zzw/RAG-Team6-score-v4/autodl
mkdir -p /root/autodl-fs/zzw/logs
nohup bash run_score_v4.sh check /root/autodl-fs/zzw/configs/new-server.json \
  > /root/autodl-fs/zzw/logs/score-v4.log 2>&1 < /dev/null &
```

The wrapper creates configs/score-v4.json adjacent to work_dir, then tune and confirmation ONLY. Existing data splits and corpus are reused. Original .venv is used without installing packages. New outputs: work_dir/optimization/score-v4/{baseline,hybrid_reasoned,hybrid_bge}/ and selection.json, confirmation.json. hybrid_bge exists only if explicitly configured before the first run.

If BGE weights already exist, prefix the FIRST command with `env BGE_PATH=/root/autodl-fs/zzw/models/bge-reranker-v2-m3 BGE_DEVICE=cuda:0`. Do not add BGE halfway through an experiment; the wrapper rejects changed config. Use CPU explicitly when GPU memory is insufficient, but measure extra latency. Reranker and Qwen can compete for GPU compute even when memory fits.

To download optional BGE weights yourself (not required for lexical test):

```bash
/root/autodl-tmp/zzw/RAG-Team6/autodl/.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-reranker-v2-m3',
    local_dir='/root/autodl-fs/zzw/models/bge-reranker-v2-m3',
    cache_dir='/root/autodl-fs/zzw/cache/huggingface',
    allow_patterns=['config.json', 'model.safetensors', 'tokenizer*',
                    'sentencepiece*', 'special_tokens_map.json', '*.model'])
PY
```

[Official BGE model and Transformers scoring contract](https://huggingface.co/BAAI/bge-reranker-v2-m3).

After inspecting validation, `bash run_score_v4.sh predict BASE_CONFIG` runs a fresh32 paired HF batching pilot and full test only if it passes. Keep the same BGE_PATH/BGE_DEVICE on every stage. Default batch8; BATCH_SIZE16 and MAX_BATCH_TOKENS294912 are optional, require their own pilot. One Qwen plus optional one BGE across all threads, not one model per thread. Reranker calls serialize for safety. New test.csv is under the selected v4 policy; never splice old v2 predictions.

## Validation limits

Local tests cover real SQLite title queries, query year/Unicode handling, duplicate-free RRF, independent-hop survival, local/global candidate reranking, profile contracts, and single reranker loading/shared workers with synthetic fixtures. These do not measure factual accuracy, real BGE throughput, GPU memory, or Kaggle performance. Run check before committing to a 12,576-question prediction.
