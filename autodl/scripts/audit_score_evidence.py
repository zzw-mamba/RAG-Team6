"""Offline label-based diagnosis ONLY. Never called from model inference."""
import argparse
from collections import Counter
from pathlib import Path

from autodl_rag.core import load_config, normalized, records, save_json
from autodl_rag.data import prepared


def contains(gold, docs):
    text = ' ' + normalized(' '.join(d['title'] + ' ' + d['text'] for d in docs)) + ' '
    return any(' ' + normalized(g) + ' ' in text for g in gold)


def audit(cfg, traces, split):
    inputs = {str(r['id']): r for r in records(prepared(cfg) / f'{split}.jsonl')}
    counts, failures = Counter(), []
    for trace in records(traces):
        row = inputs[str(trace['id'])]
        gold = row['answers']
        if all(normalized(a) in ('yes', 'no') for a in gold):
            continue
        local, visible = row['ctxs'], trace['initial_documents']
        a, b = contains(gold, local), contains(gold, visible)
        correct = any(normalized(trace['final']['answer']) == normalized(g) for g in gold)
        counts['entity_rows'] += 1
        counts['gold_in_full_local'] += a
        counts['gold_in_visible'] += b
        counts['local_gold_lost'] += a and not b
        counts['global_gold_added'] += b and not a
        counts['wrong_despite_visible_gold'] += b and not correct
        counts['correct'] += correct
        if not correct:
            failures.append({'id': row['id'], 'question': row['question'],
                             'gold_OFFLINE_ONLY': gold, 'prediction': trace['final']['answer'],
                             'gold_in_full_local': a, 'gold_in_visible': b,
                             'local_contexts': local, 'visible_contexts': visible})
    return {'counts': dict(counts), 'errors_OFFLINE_ONLY': failures,
            'note': 'Offline annotations only; never use this output as prompts, retrieval or training examples. Gold string matching is a proxy, not proof of a full reasoning chain.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--traces', required=True)
    p.add_argument('--split', choices=('tune', 'confirmation'), default='confirmation')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = audit(load_config(args.config), Path(args.traces), args.split)
    save_json(args.output, result)
    print(result['counts'], flush=True)
    print(f'Offline report: {args.output}', flush=True)


if __name__ == '__main__':
    main()
