"""Accept complete executed worked examples after independent byte-bound review."""
from __future__ import annotations
import argparse
import hashlib
import json
from example_evidence import ROOT, validate_example_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('submission')
    parser.add_argument('review')
    parser.add_argument('--ids', nargs='+')
    args = parser.parse_args()
    submission = json.loads((ROOT / args.submission).read_text())
    reviews = json.loads((ROOT / args.review).read_text())
    ids = {r['id'] for r in reviews['entries'] if r.get('accepted') is True}
    if args.ids:
        assert set(args.ids) <= ids
        ids = set(args.ids)
    path = ROOT / 'coverage/example_reproductions.json'
    overlay = json.loads(path.read_text())
    indexed = {e['id']:e for e in overlay['entries']}
    changed = []
    for raw in submission['entries']:
        if raw['id'] not in ids: continue
        e = dict(raw)
        e.pop('source_url', None)
        e.update(author=submission['author'], status='verified', verification_path=args.review)
        for field in ('solution', 'code', 'results'):
            e[f'{field}_sha256'] = hashlib.sha256((ROOT / e[f'{field}_path']).read_bytes()).hexdigest()
        validate_example_evidence(e)
        indexed[e['id']] = e
        changed.append(e['id'])
    assert set(changed) == ids
    overlay['entries'] = sorted(indexed.values(), key=lambda e:e['id'])
    path.write_text(json.dumps(overlay,indent=2,ensure_ascii=False)+'\n')
    print('Registered independently verified reproductions: ' + ', '.join(changed))


if __name__ == '__main__': main()
