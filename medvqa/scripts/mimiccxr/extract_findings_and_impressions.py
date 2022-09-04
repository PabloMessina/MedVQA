import os
from tqdm import tqdm
from medvqa.datasets.mimiccxr.preprocessing import (
    extract_report_and_patient_background,
    get_reports_txt_paths
)
from medvqa.utils.files import make_dirs_in_filepath
from medvqa.utils.common import CACHE_DIR

if __name__ == '__main__':

    print('Loading MIMIC-CXR\'s report file paths ...')
    report_file_paths = get_reports_txt_paths()

    n_failed_paths = 0
    save_path = os.path.join(CACHE_DIR, 'mimiccxr', 'findings+impression.txt')
    print(f'Saving findings + impressions to {save_path} ...')
    make_dirs_in_filepath(save_path)
    with open(save_path, 'w') as f:
        for path in tqdm(report_file_paths):
            report = extract_report_and_patient_background(path.as_posix())['report']
            if report:
                f.write(f'{report}\n')
            else:
                n_failed_paths += 1
    
    print(f'Num failed paths = {n_failed_paths}')
    print('Done!')