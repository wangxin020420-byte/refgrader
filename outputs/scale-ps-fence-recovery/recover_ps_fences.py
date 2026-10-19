"""Offline, auditable Ps parsing adapter. Raw responses and original runner stay unchanged."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalize_and_parse(raw, request, strict_parse):
    try:
        return strict_parse(raw, request), None
    except json.JSONDecodeError:
        content = raw['choices'][0]['message']['content']
        if not isinstance(content, str):
            raise
        match = re.fullmatch(r'```(?:json)?\s*\n(.*?)\n\s*```',
                             content.strip(), re.DOTALL | re.IGNORECASE)
        if not match:
            raise
        normalized = match.group(1).strip()
        trial = copy.deepcopy(raw)
        trial['choices'][0]['message']['content'] = normalized
        records = strict_parse(trial, request)
        return records, {
            'request_id': request['request_id'],
            'operation': 'remove_complete_outer_markdown_fence_only',
            'original_content_sha256': digest(content.encode('utf8')),
            'normalized_content_sha256': digest(normalized.encode('utf8')),
            'recovered_candidates': len(records),
            'answer_text_modified': False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    runner = root / 'control/scale-paper-ps/ps.py'
    spec = importlib.util.spec_from_file_location('original_scale_ps', runner)
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    run = root / 'paper_method_v1/runs/ps_glm_v1'
    audit = run / 'fence_recovery_v1'
    strict_parse = ps.parse_response

    def adapter(raw, request):
        records, entry = normalize_and_parse(raw, request, strict_parse)
        if entry is not None:
            rid = request['request_id']
            response = run / 'responses' / (rid + '.json')
            entry['raw_response_sha256'] = digest(response.read_bytes())
            entry['original_runner_sha256'] = digest(runner.read_bytes())
            entry['recovery_script_sha256'] = digest(Path(__file__).read_bytes())
            # Original execute holds process.lock during these writes and aggregation.
            ps.save(audit, {rid + '.json': ps.enc(entry)})
        return records

    def no_network(*args, **kwargs):
        raise RuntimeError('Network is disabled in offline fence recovery')

    ps.parse_response = adapter
    ps.urllib.request.urlopen = no_network
    # Original runner verifies plan/script hashes, takes the process lock,
    # validates all variants and regenerates the same run's derived outputs.
    ps.execute(root, offline=True, limit=None)
    print('OFFLINE_FENCE_RECOVERY_FINISHED; raw responses unchanged; API calls=0')
    print('Recovery audit:', audit)


if __name__ == '__main__':
    main()
