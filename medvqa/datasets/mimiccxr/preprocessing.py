import re
import os
from pathlib import Path
from medvqa.datasets.mimiccxr import MIMICCXR_DATASET_DIR

_re_header = re.compile(r'[A-Z]+( +[A-Z]+)*?:')

def extract_findings_and_impression(report_path, debug=False):
    with open(report_path) as f:
        text = f.read()
    if debug:
        print(text)
    text = text.replace('_', '')
    spans = [i.span() for i in _re_header.finditer(text)]
    report = ''
    for i, span in enumerate(spans):
        match = text[span[0]:span[1]]
        if (len(match) > 20 or match == 'FINDINGS:' or match == 'IMPRESSION:' or match == 'CONCLUSION:'):
            if i+1 == len(spans):
                x = text[span[1]:]
            else:
                x = text[span[1]:spans[i+1][0]]
            x = ' '.join(x.split())
            if x:
                if report:
                    report += ' ' if report[-1] == '.' else '. '
                report += x
    if not report:
        for part in re.split('\s*\n\s*\n\s*', text):
            part = ' '.join(part.split())
            if len(part) > 150:
                report = part
    return report

def report_paths_generator():
    for x in range(10, 20):
        for filepath in Path(os.path.join(MIMICCXR_DATASET_DIR, f'files/p{x}/')).rglob("s*.txt"):
            yield filepath