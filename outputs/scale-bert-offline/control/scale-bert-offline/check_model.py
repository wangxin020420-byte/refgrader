"""Verify the pinned offline model, then optionally run a CPU forward check."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / 'sources/bert-offline-manifest.json').read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        target = (root / relative).resolve()
        target.relative_to(root)
        if not target.is_file() or sha256(target) != expected:
            raise ValueError('Missing or changed file: ' + relative)
    lock = json.loads((root / 'sources/bert-base-chinese.lock.json').read_text(encoding='utf-8'))
    if lock != {'repo_id': manifest['repo_id'], 'revision': manifest['revision']}:
        raise ValueError('Model lock does not match offline manifest')
    print('OFFLINE_FILES_OK', flush=True)
    if args.verify_only:
        return

    # Set these before importing the model libraries: no server network or GPU needed.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    torch.set_num_threads(2)
    model_dir = root / 'models/bert-base-chinese' / lock['revision']
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True,
                                      trust_remote_code=False, use_safetensors=True)
    model.eval()
    tokens = tokenizer('\u4ec0\u4e48\u662f\u64cd\u4f5c\u7cfb\u7edf\uff1f',
                       '\u64cd\u4f5c\u7cfb\u7edf\u7ba1\u7406\u8ba1\u7b97\u673a\u8d44\u6e90\u3002',
                       return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        output = model(**tokens).last_hidden_state
    if not torch.isfinite(output).all().item():
        raise ValueError('Non-finite model output')
    if list(output.shape) != [1, tokens['input_ids'].shape[1], model.config.hidden_size]:
        raise ValueError('Unexpected output dimensions')
    report = {
        'status': 'MODEL_CHECK_OK', 'python': sys.version,
        'python_executable': sys.executable, 'torch': torch.__version__,
        'torch_cuda_build': torch.version.cuda, 'transformers': transformers.__version__,
        'repo_id': lock['repo_id'], 'revision': lock['revision'],
        'model_dir': str(model_dir), 'device': str(next(model.parameters()).device),
        'output_shape': list(output.shape), 'offline': True, 'training_started': False,
        'manifest_sha256': sha256(root / 'sources/bert-offline-manifest.json'),
    }
    reports = root / 'reports'
    reports.mkdir(exist_ok=True)
    report_path = reports / 'bert_model_check.json'
    payload = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    temp = reports / 'bert_model_check.json.tmp'
    temp.write_text(payload, encoding='utf-8')
    temp.replace(report_path)
    print(payload)


if __name__ == '__main__':
    main()
