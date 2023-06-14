from medvqa.utils.files import save_json
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR, get_reports_txt_paths
from medvqa.datasets.mimiccxr.preprocessing import extract_background_findings_and_impression
from medvqa.utils.common import get_timestamp

from tqdm import tqdm
import os

if __name__ == '__main__':
    
    # Get report paths    
    report_paths = get_reports_txt_paths()

    # Extract background findings and impression
    print(f'Processing {len(report_paths)} reports from MIMIC-CXR...')
    results = [None] * len(report_paths)
    for i, report_path in tqdm(enumerate(report_paths), total=len(report_paths), mininterval=2.0):
        results[i] = extract_background_findings_and_impression(report_path)
        results[i]['path'] = report_path # Add report path to results
    
    # Save results
    save_path = os.path.join(MIMICCXR_CACHE_DIR, f'background_findings_and_impression_{get_timestamp()}.json')
    save_json(results, save_path)
    print(f'Saved results to {save_path}')