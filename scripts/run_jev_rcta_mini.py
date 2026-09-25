"""Bounded concurrent Mini execution; frozen inference and resumable calls."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
import threading
import time

import evaluate as ev
import evaluate_jev_rcta as mini
import jev_rcta as method
import run_jev_rcta_hosted_full as shared


class CaseStop:
    """A case failure cancels that case; quota/global stops cancel every case."""
    def __init__(self, global_stop):
        self.local = threading.Event()
        self.global_stop = global_stop

    def set(self):
        self.local.set()

    def is_set(self):
        return self.local.is_set() or self.global_stop.is_set()

    def wait(self, seconds):
        return self.is_set() or self.global_stop.wait(seconds)


def isolated_block(error):
    return isinstance(error, RuntimeError) and str(error).startswith('HTTP 403:') and 'Cloudflare' in str(error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--skip-id', action='append', default=[], help='Record an unavailable case and continue other cases')
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError('Use 1..4 workers')
    rows, manifest = ev.load_data()
    skipped = sorted(set(args.skip_id))
    if not set(skipped) <= {r['question_ID'] for r in rows}:
        raise ValueError('Unknown skipped case ID')
    out = ev.ROOT / args.output
    config = mini.prepare(out, manifest)
    selected = {'selection': 'mini', 'question_ids': [r['question_ID'] for r in rows]}
    if (out / 'selection.json').exists() and json.loads((out / 'selection.json').read_text()) != selected:
        raise ValueError('Sample selection changed')
    mini.write_json(out / 'selection.json', selected)
    if (out / 'balance_stop.json').exists():
        raise ValueError('Persistent quota stop; no requests allowed')
    key = os.environ.get('TYPESAFE_API_KEY') or os.environ.get('JEV_API_KEY')
    if not key:
        for line in (ev.ROOT / '.env').read_text().splitlines():
            if line.startswith(('TYPESAFE_API_KEY=', 'JEV_API_KEY=')):
                key = line.split('=', 1)[1].strip().strip('\"\'')
    if not key:
        raise ValueError('Missing private API key')
    shared.ENDPOINT = ev.ENDPOINT
    stop = threading.Event()
    predictions, failures = {}, []
    started = time.time()
    execution = {'endpoint': ev.ENDPOINT, 'workers': args.workers, 'started_unix': started,
                 'skipped_ids': skipped,
                 'runner_sha256': ev.sha(__import__('pathlib').Path(__file__).read_bytes()),
                 'shared_runner_sha256': ev.sha((ev.ROOT / 'scripts/run_jev_rcta_hosted_full.py').read_bytes()),
                 'note': 'Frozen inference; exact cached requests reused. Record isolated Cloudflare 403 cases as missing. Stop globally on quota or other errors.'}
    mini.write_json(out / 'execution.json', execution)

    def progress(status='running'):
        mini.write_json(out / 'progress.json', {'status': status, 'completed_n': len(predictions),
                        'skipped_ids': skipped,
                        'requested_n': len(rows), 'completed_ids': list(predictions),
                        'failed_ids': failures, 'updated_unix': time.time()})

    def accept(row):
        qid = row['question_ID']
        try:
            predictions[qid] = shared.run_case(row, out, key, stop)
            print('MINI %d/%d completed' % (len(predictions), len(rows)), flush=True)
        except Exception as error:
            stop.set()
            failures.append(qid)
            print('STOP: ' + type(error).__name__, flush=True)
        progress()

    # Valid end-to-end output gate, without consulting reference accuracy.
    smoke = [min((r for r in rows if r['source'] == source),
                 key=lambda r: (len(ev.dumps(r['history']).encode()), r['question_ID']))
             for source in sorted({r['source'] for r in rows})]
    for row in smoke:
        if row['question_ID'] in skipped:
            continue
        accept(row)
        if stop.is_set():
            break
    if not stop.is_set():
        mini.write_json(out / 'preflight.json', {'passed': True, 'question_ids': list(predictions)})
        remaining = iter(r for r in rows if r['question_ID'] not in predictions and r['question_ID'] not in skipped)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {}
            def fill():
                while len(pending) < args.workers and not stop.is_set():
                    row = next(remaining, None)
                    if row is None:
                        break
                    pending[pool.submit(shared.run_case, row, out, key, CaseStop(stop))] = row
            fill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    row = pending.pop(future)
                    try:
                        predictions[row['question_ID']] = future.result()
                        print('MINI %d/%d completed' % (len(predictions), len(rows)), flush=True)
                    except Exception as error:
                        if isolated_block(error):
                            skipped.append(row['question_ID'])
                            skipped.sort()
                            print('SKIP blocked case: ' + row['question_ID'], flush=True)
                        else:
                            stop.set()
                            failures.append(row['question_ID'])
                            print('STOP: ' + type(error).__name__, flush=True)
                progress()
                fill()
    status = 'error' if stop.is_set() else 'complete_with_skips' if skipped else 'complete'
    if (out / 'balance_stop.json').exists():
        status = json.loads((out / 'balance_stop.json').read_text())['status']
    progress(status)
    account = shared.accounting(out)
    account['reported_cost_usd'] = None  # This endpoint reports tokens, not dollar charges.
    mini.write_json(out / 'accounting.json', account)
    completed = [r for r in rows if r['question_ID'] in predictions]
    if completed:
        summary = ev.summarize(completed, predictions, manifest)
        summary.pop('estimated_input_cost_usd', None)
        summary.update(protocol=config, status=status, expected_n=len(rows), accounting=account,
                       cost_note='API reports token usage, not actual dollar charges.',
                       run_wall_seconds=time.time()-started)
        summary['candidate_recall'] = {field: sum(r['mistake_step'] in predictions[r['question_ID']][field]
            for r in completed) / len(completed) for field in
            ('recall_candidates', 'trace_seeds', 'expanded_candidates', 'final_candidates')}
        mini.write_json(out / 'summary.json', summary)
    execution.update(ended_unix=time.time(), status=status)
    mini.write_json(out / 'execution.json', execution)
    print(json.dumps({'status': status, 'completed': len(predictions), 'accounting': account}), flush=True)
    if stop.is_set():
        raise SystemExit(2)


if __name__ == '__main__':
    main()
