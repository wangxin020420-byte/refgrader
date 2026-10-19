import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import recover_proxy_json as recovery


def content(number):
    if number == '515':
        return '{"proxy_answer": "第一行。\n第二行。"}'
    return '{"proxy_answer": "前文' + recovery.PHRASES[number] + '后文。\\n末尾。"}'


class RecoveryTests(unittest.TestCase):
    def test_quotes_and_newlines_preserved(self):
        for number in recovery.PHRASES:
            answer = json.loads(recovery.recover_content(content(number), number))['proxy_answer']
            expected = ('第一行。\n第二行。' if number == '515' else
                        '前文' + recovery.PHRASES[number] + '后文。\n末尾。')
            self.assertEqual(answer, expected)

    def test_unknown_damage_rejected(self):
        for malformed in ('{"proxy_answer":"x", "extra":1}',
                          '{"proxy_answer":"unfinished',
                          '{"proxy_answer":"bad\\q"}'):
            with self.assertRaises(ValueError):
                recovery.recover_content(malformed, '515')
        with self.assertRaises(ValueError):
            recovery.recover_content('{"proxy_answer":"other"}', '62')

    def test_offline_lifecycle_with_original_runner(self):
        # Windows test host: mock only the Linux file-lock API, not the runner.
        fake_fcntl = types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=lambda *args: None)
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {'fcntl': fake_fcntl}):
            root = Path(directory)
            script = root / 'control/scale-paper-proxy/proxy.py'
            script.parent.mkdir(parents=True)
            shutil.copy2(Path(__file__).with_name('proxy.py'), script)
            runner = recovery.load_runner(root)
            sources = [{'sample_id': 'ADS:9e1a866d58ed:train:' + n, 'question_id': 'q',
                        'q': 'question', 'a': 'original', 'r': [], 'l': 3}
                       for n in recovery.PHRASES]
            # Include a normal successful response to check it remains unchanged.
            sources.append(dict(sources[0], sample_id='ADS:9e1a866d58ed:train:999'))
            requests, blanks = runner.make_plan(sources)
            plan = root / 'paper_method_v1/artifacts/proxy_inputs_v1'
            cfg = {'script_sha256': runner.sha(script)}
            files = {'config.json': runner.enc(cfg), 'training_sources.json': runner.enc(sources),
                     'requests.jsonl': runner.jl(requests), 'blank_sources.json': runner.enc(blanks)}
            for name, data in files.items():
                runner.save(plan / name, data)
            runner.save(plan / 'summary.json', runner.enc({'files_sha256': {
                name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}))
            run = root / 'paper_method_v1/runs/proxy_glm_v1'
            for req in requests:
                n = req['sample_id'].rsplit(':', 1)[-1]
                text = content(n) if n in recovery.PHRASES else '{"proxy_answer":"valid"}'
                runner.save(run / 'responses' / (req['request_id'] + '.json'), runner.enc({
                    'choices': [{'finish_reason': 'stop', 'message': {'content': text}}]}))
            before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with contextlib.redirect_stdout(io.StringIO()):
                recovery.execute(root, apply=False)
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob('*') if p.is_file()})
            with contextlib.redirect_stdout(io.StringIO()):
                recovery.execute(root, apply=True)
                recovery.execute(root, apply=True)
            summary = runner.read(run / 'summary.json')
            self.assertEqual(summary['valid_requests'], 5)
            self.assertEqual(summary['new_api_calls_this_invocation'], 0)
            self.assertEqual(summary['errors'], [])
            self.assertFalse(summary['ready_for_annotator_training'])
            for p, data in before.items():
                self.assertEqual(p.read_bytes(), data)
            self.assertEqual(len(runner.read(run / 'json_recovery_audit_v1.json')['repairs']), 4)


if __name__ == '__main__':
    unittest.main()
