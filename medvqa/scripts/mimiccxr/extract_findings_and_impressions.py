import os
from tqdm import tqdm
from medvqa.datasets.mimiccxr.preprocessing import (
    extract_findings_and_impression,
    report_paths_generator
)
from medvqa.utils.files import make_dirs_in_filepath
from medvqa.utils.common import CACHE_DIR

if __name__ == '__main__':
    
    report_file_paths = [None] * 300000

    print('Loading MIMIC-CXR\'s report file paths ...')
    for i, filepath in tqdm(enumerate(report_paths_generator())):
        report_file_paths[i] = filepath
    assert report_file_paths[i] is not None
    assert report_file_paths[i+1] is None
    report_file_paths = report_file_paths[:i+1]

    failed_paths = []
    save_path = os.path.join(CACHE_DIR, 'mimiccxr', 'findings+impression.txt')
    print(f'Saving findings + impressions to {save_path} ...')
    make_dirs_in_filepath(save_path)
    with open(save_path, 'w') as f:
        for path in tqdm(report_file_paths):
            report = extract_findings_and_impression(path.as_posix())
            if report:
                f.write(f'{report}\n')
            else:
                failed_paths.append(path)
    
    print(f'Num failed paths = {len(failed_paths)}')
    print('Done!')