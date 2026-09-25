import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_jev_rcta_mini as runner


class MiniRunnerTests(unittest.TestCase):
    def exercise(self, failure=None, skipped=None, blocked=None):
        rows = [{'question_ID': '%s%d' % (s, i), 'source': s,
                 'history': [{'step': 0, 'name': 'Worker', 'content': 'x' * (i+1)}],
                 'mistake_step': 0, 'mistake_agent': 'Worker'}
                for s in 'abcde' for i in range(2)]
        seen = []
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'scripts').mkdir()
            (root / 'scripts/run_jev_rcta_hosted_full.py').write_text('fixture')
            def prepare(out, manifest):
                (out / 'calls').mkdir(parents=True)
                return {}
            def run(row, out, key, stop):
                seen.append(row['question_ID'])
                if row['question_ID'] == blocked:
                    stop.set()
                    raise RuntimeError('HTTP 403: Attention Required! | Cloudflare')
                if row['question_ID'] == failure:
                    stop.set()
                    runner.mini.write_json(out / 'balance_stop.json', {'status': 'paused_insufficient_balance'})
                    raise RuntimeError('quota exhausted')
                return {'predicted_role': 'Worker', 'predicted_step': 0, 'method': 'fixture',
                        **{f: [0] for f in ('recall_candidates', 'trace_seeds', 'expanded_candidates', 'final_candidates')}}
            with patch.object(runner.ev, 'ROOT', root), \
                 patch.object(runner.ev, 'load_data', return_value=(rows, {'revision': 'test'})), \
                 patch.object(runner.mini, 'prepare', side_effect=prepare), \
                 patch.object(runner.shared, 'run_case', side_effect=run), \
                 patch.dict(runner.os.environ, {'TYPESAFE_API_KEY': 'test-private'}), \
                 patch.object(sys, 'argv', ['runner', '--output', 'out', '--workers', '1'] + (['--skip-id', skipped] if skipped else [])), \
                 contextlib.redirect_stdout(io.StringIO()):
                if failure:
                    with self.assertRaises(SystemExit):
                        runner.main()
                else:
                    runner.main()
            return seen, json.loads((root/'out/progress.json').read_text())

    def test_all_mini_rows_once_after_valid_preflight(self):
        seen, progress = self.exercise()
        self.assertEqual(seen[:5], [s+'0' for s in 'abcde'])
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(progress['completed_n'], 10)
        self.assertEqual(progress['status'], 'complete')

    def test_preflight_quota_prevents_main_run(self):
        seen, progress = self.exercise('a0')
        self.assertEqual(seen, ['a0'])
        self.assertEqual(progress['status'], 'paused_insufficient_balance')

    def test_main_quota_prevents_scheduling_more_cases(self):
        seen, progress = self.exercise('a1')
        self.assertEqual(len(seen), 6)
        self.assertEqual(progress['completed_n'], 5)
        self.assertEqual(progress['status'], 'paused_insufficient_balance')

    def test_skipped_case_is_never_called_or_counted_as_completed(self):
        seen, progress = self.exercise(skipped='a1')
        self.assertNotIn('a1', seen)
        self.assertEqual(progress['skipped_ids'], ['a1'])
        self.assertEqual(progress['completed_n'], 9)
        self.assertEqual(progress['requested_n'], 10)
        self.assertEqual(progress['status'], 'complete_with_skips')

    def test_isolated_cloudflare_block_preserves_other_cases(self):
        seen, progress = self.exercise(blocked='a1')
        self.assertEqual(len(seen),10)
        self.assertEqual(progress['completed_n'],9)
        self.assertEqual(progress['skipped_ids'],['a1'])
        self.assertEqual(progress['status'],'complete_with_skips')

    def test_case_stop_is_local_but_global_stop_reaches_all_cases(self):
        global_stop = runner.threading.Event()
        first, second = runner.CaseStop(global_stop), runner.CaseStop(global_stop)
        first.set()
        self.assertTrue(first.is_set())
        self.assertFalse(second.is_set())
        global_stop.set()
        self.assertTrue(second.is_set())
        self.assertTrue(second.wait(0))


if __name__ == '__main__':
    unittest.main()
