import json
from pathlib import Path
import tempfile
import unittest
import audit_visual_transcriptions_v1 as m


class Tests(unittest.TestCase):
    def test_flag_is_not_automatic_failure(self):
        t = {'text': '如图所示：A指向B，B指向C。', 'issues': []}
        self.assertEqual(m.flags('如图所示', t), ['VISUAL_MARKER_REMAINS_CHECK_EXPANSION'])
        self.assertTrue(m.TRIGGER.search('见下表'))
        self.assertFalse(m.TRIGGER.search('没有图形'))

    def test_offline_inventory_preserves_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inputs, run = root / 'inputs', root / 'run'
            inputs.mkdir()
            run.mkdir()
            image = root / 'image.jpg'
            image.write_bytes(b'fixture-image')
            records = [dict(sample_id=str(i), question_id='q', answer=t, blank_passthrough=not t,
                            student_image=str(image), text_review_required=False)
                       for i,t in enumerate(['如图所示', '见表', ''])]
            data = {'train.inputs.jsonl':records, 'validation.inputs.jsonl':[], 'test.inputs.jsonl':[],
                    'questions.jsonl':[dict(question_id='q', question='q', references=['reference'])]}
            hashes={}
            for name, rr in data.items():
                raw=''.join(json.dumps(r)+'\n' for r in rr).encode()
                (inputs/name).write_bytes(raw)
                hashes[name]=m.digest(raw)
            (inputs/'manifest.json').write_bytes(m.encoded({'files_sha256':hashes}))
            # Labels intentionally invalid; reading them must not be necessary.
            (inputs/'train.labels.jsonl').write_text('NOT_JSON')
            uid='answer:0'
            unit=dict(unit_id=uid, kind='student_answer', image=str(image), image_sha256=m.digest(image.read_bytes()), original_text='如图所示')
            p=dict(input_directory=str(inputs),input_manifest_sha256=m.digest((inputs/'manifest.json').read_bytes()),
                   base_url='https://open.bigmodel.cn/api/coding/paas/v4/',model='glm-4.6v',units=[unit],missing_visual_context=[])
            (run/'plan.json').write_bytes(m.encoded(p))
            folder=run/'responses'/m.digest(uid.encode())
            folder.mkdir(parents=True)
            t={'text':'如图所示：A指向B','issues':[]}
            response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(t)}}]}
            (folder/'1.response.json').write_bytes(m.encoded(response))
            s=dict(unit_id=uid,plan_sha256=m.digest(m.encoded(p)),response_file='1.response.json',
                   response_sha256=m.digest((folder/'1.response.json').read_bytes()),transcription=t)
            (folder/'accepted.json').write_bytes(m.encoded(s))
            before={str(f):f.read_bytes() for pth in (inputs,run) for f in pth.rglob('*') if f.is_file()}
            result=m.audit(inputs,run,root/'audit',enforce_counts=False)
            self.assertEqual(result['accepted_checked'],1)
            self.assertEqual(result['uncovered_visual_cues'],1)
            self.assertEqual(result['blank_counts']['train'],1)
            self.assertFalse(result['ready_for_training'])
            self.assertTrue(all(Path(f).read_bytes()==raw for f,raw in before.items()))
            with self.assertRaises(FileExistsError):
                m.audit(inputs,run,root/'audit',enforce_counts=False)


if __name__=='__main__':
    unittest.main()
