"""Offline integrity/dependency preflight; never calls an API."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

def main():
    manifest=ROOT/'BUNDLE_SHA256.json'
    if not manifest.exists():
        raise SystemExit('Missing BUNDLE_SHA256.json: use the complete deployment package')
    for name,expected in json.loads(manifest.read_text(encoding='utf8')).items():
        p=ROOT/name
        if not p.resolve().is_relative_to(ROOT.resolve()) or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
            raise SystemExit('Bundle mismatch: '+name)
    for module in ['numpy','scipy','PIL','openai','json_repair','sklearn','filelock','main_pipeline']:
        importlib.import_module(module)
    from model_runtime import runtime_model_config
    assert runtime_model_config(text_provider='glm53')['text_model']=='glm-5.3'
    print('BUNDLE_AND_IMPORTS_OK_NO_API_CALLS')

if __name__=='__main__':
    main()
