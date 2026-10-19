"""Normalize legacy socket.timeout without modifying the frozen entity runner."""
import hashlib
import json
from pathlib import Path
import socket
import sys
import urllib.request

EXPECTED = {
    'entity_engine.py': '32757e1fcc5bfaa5325a16c6a57152a26819ad29ddd10bde331c3a718dcc612d',
    'run_entities.py': 'b3ebb0d5d073e604d047c5fb6c8b8f9c206ceb43dbc84851584d025f364a829d',
    'stage1_contract.py': '26c9fe7ee85fb7760e70bfa29da87533cd1266d7d147e2e60c1fa543355c919c',
}


class NormalizedResponse:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        self.response.__enter__()
        return self

    def __exit__(self, *args):
        return self.response.__exit__(*args)

    def read(self, *args, **kwargs):
        try:
            return self.response.read(*args, **kwargs)
        except socket.timeout as exc:
            raise TimeoutError(str(exc)) from exc


def adapt_urlopen(original):
    def call(*args, **kwargs):
        try:
            return NormalizedResponse(original(*args, **kwargs))
        except socket.timeout as exc:
            raise TimeoutError(str(exc)) from exc
    return call


def main():
    project = Path('/data0/E125221219/projects/scale-private')
    original = project/'control/train_entities_clean_v1'
    for name, expected in EXPECTED.items():
        actual = hashlib.sha256((original/name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Frozen runner changed: '+str(original/name))
    sys.path.insert(0, str(original))
    import run_entities as adapter
    import entity_engine as engine
    out = project/'runs/online_experiments'/adapter.EXP/'train_entities_five_runs_v1'
    if not (out/'config.json').is_file():
        raise ValueError('Existing entity run missing')
    audit = dict(
        status='TIMEOUT_COMPATIBILITY_RECOVERY',
        original_runner_sha256=EXPECTED,
        recovery_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        policy='Normalize socket.timeout during connection and response read to builtin TimeoutError; existing retry policy handles it.',
        attempts_per_unit=8, timeout_seconds=900,
        request_payload_changed=False, original_scripts_modified=False,
        accepted_cache_reused=True, labels_modified=False)
    engine.write(out/'timeout_recovery_v1.json', engine.enc(audit))
    urllib.request.urlopen = adapt_urlopen(urllib.request.urlopen)
    sys.argv = [str(original/'run_entities.py'), 'run', '--root', str(project)]
    return adapter.main()


if __name__ == '__main__':
    raise SystemExit(main())
