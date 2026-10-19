"""Change only the startup gate; migrate config hashes for unstarted runs."""
import argparse
import hashlib
import json
from pathlib import Path

OLD_SHA = '47cc9ba048398bdfec1f6a00373af8da369ba7c3d9108340230947900dbea1ab'


def digest(content):
    return hashlib.sha256(content).hexdigest()


def keep(path, content):
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError('Backup differs: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(content)


def replace(path, content):
    tmp = path.with_name(path.name + '.gpu-gate.tmp')
    tmp.write_bytes(content)
    tmp.replace(path)


def patch(root):
    probe = root / 'reports/gpu2_memory_probe_1789354767848699240.json'
    report = json.loads(probe.read_text(encoding='utf8'))
    if (report.get('status'), report.get('microbatch'), report.get('effective_batch'),
        report.get('sequence_length'), report.get('optimizer_steps')) != ('GPU_MEMORY_PROBE_OK', 2, 16, 256, 2):
        raise ValueError('Unexpected memory probe')
    if not 0 < report['torch_peak_reserved_mib'] <= 3072:
        raise ValueError('Memory peak exceeds this patch assumption')
    script = root / 'control/scale-abc-multiseed/abc_experiment.py'
    backup = script.with_name('abc_experiment.before_gpu_gate.py')
    old = backup.read_bytes() if backup.exists() else script.read_bytes()
    if digest(old) != OLD_SHA:
        raise ValueError('Trainer version differs; stop rather than patch unknown code')
    new = old.replace(b'if free < 8 * 1024 ** 3:', b'if free < 4 * 1024 ** 3:').replace(
        b'Less than 8 GiB free;', b'Less than 4 GiB free;')
    if new == old or script.read_bytes() not in (old, new):
        raise ValueError('Unexpected current script')
    new_sha = digest(new)
    changes = []
    for seed in (43, 44, 45, 46):
        for group in 'ABC':
            run = root / 'runs/abc_followup_v1' / ('%s_seed%d' % (group, seed))
            if not run.exists():
                continue
            allowed = {'config.json', 'config.before_gpu_gate.json', 'config.json.gpu-gate.tmp'}
            if any(p.name not in allowed for p in run.iterdir()):
                raise ValueError('Run has training artifacts or lock; migration refused: ' + str(run))
            cfg_path = run / 'config.json'
            if not cfg_path.exists():
                continue
            raw = cfg_path.read_bytes()
            cfg = json.loads(raw)
            if cfg['seed'] != seed or cfg['group'] != group or cfg['script_sha256'] not in (OLD_SHA, new_sha):
                raise ValueError('Config provenance mismatch: ' + str(run))
            if cfg['script_sha256'] == OLD_SHA:
                cfg['script_sha256'] = new_sha
                content = (json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True)+'\n').encode('utf8')
                cfg_backup = run / 'config.before_gpu_gate.json'
                if cfg_backup.exists() and cfg_backup.read_bytes() != raw:
                    raise ValueError('Config backup differs')
                changes.append((cfg_path, cfg_backup, raw, content))
    audit = {'old_script_sha256': OLD_SHA, 'new_script_sha256': new_sha,
             'probe_sha256': digest(probe.read_bytes()), 'startup_free_gib_before': 8,
             'startup_free_gib_after': 4, 'training_hyperparameters_changed': False,
             'migration_scope': 'unstarted seeds43-46; seed42 unchanged'}
    audit_path = root / 'reports/gpu_gate_patch_v1.json'
    audit_bytes = (json.dumps(audit, indent=2, sort_keys=True)+'\n').encode('utf8')
    if audit_path.exists() and audit_path.read_bytes() != audit_bytes:
        raise ValueError('Existing patch audit differs')
    keep(backup, old)
    for path, cfg_backup, raw, content in changes:
        keep(cfg_backup, raw)
        replace(path, content)
    replace(script, new)
    keep(audit_path, audit_bytes)
    print('GPU_GATE_PATCH_OK: 4 GiB startup threshold; hyperparameters unchanged.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    patch(parser.parse_args().root.resolve())
