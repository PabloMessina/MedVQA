from dotenv import load_dotenv
load_dotenv()

import os
from medvqa.datasets.iuxray import IUXRAY_REPORTS_MIN_JSON_PATH
from medvqa.utils.files import (
    load_json_file,
    make_dirs_in_filepath,
)
from medvqa.utils.common import CACHE_DIR

if __name__ == '__main__':

    # Load json with reports
    print(f'Loading file {IUXRAY_REPORTS_MIN_JSON_PATH} ...')
    reports = load_json_file(IUXRAY_REPORTS_MIN_JSON_PATH)

    # Extracting and merge findings and impressions
    print('Extracing and merging findings and impressions ...')
    merged = []
    for x in reports.values():
        findings = x['findings']
        impression = x['impression']
        if findings or impression:
            if findings and not impression:
                text = findings
            elif not findings and impression:
                text = impression
            else:
                if findings[-1] == '.':
                    text = findings + ' ' + impression
                else:
                    text = findings + '. ' + impression
            merged.append(text)

    # Save to file
    save_path = os.path.join(CACHE_DIR, 'iuxray', 'findings+impression.txt')
    print(f'Saving to {save_path} ...')
    make_dirs_in_filepath(save_path)
    with open(save_path, 'w') as f:
        for line in merged:
            f.write(f'{line}\n')
    
    print('done!')