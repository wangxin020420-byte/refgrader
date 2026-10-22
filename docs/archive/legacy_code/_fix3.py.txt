import re

p = 'step4_vlm_grader.py'
lines = open(p, encoding='utf-8').read().split('\n')
out = []
fixes = 0
for line in lines:
    if '"text": "\\n' in line and 'extend' in line:
        if 'q_b64' in line:
            new = re.sub(r'"text": "\\n[^"]*"', '"text": "\\n【附图】"', line)
        elif 'student_b64' in line:
            new = re.sub(r'"text": "\\n[^"]*"', '"text": "\\n【考卷】"', line)
        else:
            new = line
        if new != line:
            fixes += 1
        out.append(new)
    else:
        out.append(line)
open(p, 'w', encoding='utf-8').write('\n'.join(out))
print('label fixes:', fixes)

b = open(p, 'rb').read()
fug = '【附图】'.encode('utf-8')
kao = '【考卷】'.encode('utf-8')
print('fug_label_count:', b.count(fug))
print('kao_label_count:', b.count(kao))
