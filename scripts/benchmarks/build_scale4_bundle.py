"""Build an isolated, secret-free source and public-data deployment snapshot."""
import hashlib
import json
import shutil
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/refgrader_scale4_20260930'
PACKAGE=OUT/'refgrader-scale4'

def main():
    PACKAGE.mkdir(parents=True,exist_ok=True)
    names=['main_pipeline.py','step3_rrd_generator.py','step4_vlm_grader.py','evaluate.py',
           'model_runtime.py','calibration_utils.py','rubric_semantics.py','portable_hash.py','sample_quality.py',
           'scripts/calibrate_a3wa.py','scripts/benchmarks/run_benchmark.py','scripts/benchmarks/run_scale4.py',
           'scripts/benchmarks/scale4_data.py','scripts/benchmarks/scale4_requirements.txt',
           'scripts/benchmarks/setup_scale4.sh','scripts/benchmarks/launch_scale4.sh',
           'scripts/benchmarks/verify_scale4_bundle.py','SCALE4_REFGRADER_GUIDE.md']
    for folder,glob in [('canonicalizers','*.py'),('ocr','*.py'),('benchmark_datasets','*.py'),('prompts','*')]:
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob(glob) if p.is_file() and '__pycache__' not in p.parts]
    included=set(names)
    for name in sorted(included):
        target=PACKAGE/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes().replace(b'\r\n',b'\n'))
    source=ROOT/'tmp/refgrader_scale4_source'
    for p in source.rglob('*'):
        if p.is_file():
            target=PACKAGE/'sources/scale4_public'/p.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
            included.add(target.relative_to(PACKAGE).as_posix())
    # Explicit allowlist: never package API results, caches, environments or secrets
    # left in the staging directory by a later preflight.
    payload={name:hashlib.sha256((PACKAGE/name).read_bytes()).hexdigest()
             for name in sorted(included)}
    (PACKAGE/'BUNDLE_SHA256.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf8')
    archive=OUT/'refgrader-scale4.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(included | {'BUNDLE_SHA256.json'}):
            p=PACKAGE/name
            z.write(p,p.relative_to(OUT).as_posix())
    digest=hashlib.sha256(archive.read_bytes()).hexdigest()
    (OUT/'refgrader-scale4.zip.sha256').write_text(digest+'  refgrader-scale4.zip\n',encoding='ascii')
    print(archive)
    print('SHA256',digest,'FILES',len(payload),'BYTES',archive.stat().st_size)

if __name__=='__main__':
    main()
