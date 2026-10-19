from pathlib import Path

base = Path(__file__).parent
source = (base / 'transcribe_csbench.py').read_text(encoding='utf-8')

def replace(old, new):
    global source
    if source.count(old) != 1:
        raise ValueError('Unexpected source; refusing patch: ' + old[:70])
    source = source.replace(old, new)

replace('import os\n', 'import os\nimport re\n')
replace('    obj = json.loads(text)\n', '''    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        if not exc.msg.startswith('Invalid control character'):
            raise
        # Only allow literal control characters within JSON strings.
        # Never guess quotes, braces, formulas, or missing content.
        obj = json.loads(text, strict=False)
    if not isinstance(obj, dict):
        raise ValueError('Response must be a JSON object')
''')
replace('    fatal = None\n', '    fatal = None\n    consecutive_bad_requests = 0\n')
replace('        if accepted.exists():\n', '''        if not accepted.exists():
            # Deterministic: latest fully parseable, complete saved response.
            for old_response in sorted(folder.glob('*.response.json'), reverse=True):
                try:
                    recovered = parse_response(read(old_response))
                except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                    continue
                saved = dict(unit_id=u['unit_id'], plan_sha256=fingerprint,
                             response_file=old_response.name,
                             response_sha256=sha(old_response.read_bytes()),
                             transcription=recovered,
                             recovery_policy='v2_control_characters_only_latest_parseable',
                             semantic_review_required=True)
                write(accepted, saved)
                print('OFFLINE_RECOVERED', u['unit_id'], flush=True)
                break
        if accepted.exists():
''')
replace('                parsed = parse_response(response)\n', '                consecutive_bad_requests = 0\n                parsed = parse_response(response)\n')
replace("                write(folder / (token + '.error.json'), error)\n", '''                # Capture only service code/message, never headers or request bodies.
                body = getattr(e, 'body', None)
                if isinstance(body, dict):
                    detail = body.get('error', body)
                    if isinstance(detail, dict):
                        for field in ('code', 'message'):
                            value = detail.get(field)
                            if isinstance(value, (str, int)):
                                value = str(value)
                                for name in ('ZHIPUAI_API_KEY', 'ZHIPU_API_KEY'):
                                    secret = os.environ.get(name)
                                    if secret:
                                        value = value.replace(secret, '[REDACTED]')
                                value = re.sub(r'data:image/[^\\s"<>]+', '[IMAGE_REDACTED]', value)
                                value = re.sub(r'(?i)bearer\\s+[^\\s"<>]+', 'Bearer [REDACTED]', value)
                                error['service_' + field] = value[:1200]
                write(folder / (token + '.error.json'), error)
                print('REQUEST_ERROR', json.dumps(error, ensure_ascii=False), flush=True)
                if status == 400:
                    consecutive_bad_requests += 1
                    errors.append(error)
                    if consecutive_bad_requests >= 3:
                        fatal = error
                    # Preserve failed unit; continue independent units, no bypass.
                    break
''')
replace("                             response_sha256=sha((folder / response_file).read_bytes()), transcription=parsed)\n", "                             response_sha256=sha((folder / response_file).read_bytes()), transcription=parsed,\n                             parser_policy='v2_control_characters_only')\n")
replace("                   model=p['model'], plan_sha256=fingerprint, labels_read=False, training_started=False,\n", "                   model=p['model'], plan_sha256=fingerprint, labels_read=False, training_started=False,\n                   runner_sha256=sha(Path(__file__).read_bytes()), parser_policy='v2_control_characters_only',\n")
dest = base / 'transcribe_csbench_coding_v2.py'
dest.write_text(source, encoding='utf-8', newline='\n')
print(dest)
