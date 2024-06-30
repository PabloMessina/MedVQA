from medvqa.utils.common import TMP_DIR, get_timestamp
import os
import subprocess
import pandas as pd
import time

RADCLIQ_FOLDER = os.environ['RADCLIQ_FOLDER']
RADCLIQ_PYTHON = os.environ['RADCLIQ_PYTHON']
RADCLIQ_CONDA_ENV = os.environ['RADCLIQ_CONDA_ENV']
TMP_FOLDER = os.path.join(TMP_DIR, 'radcliq')

def _get_custom_env(device_id = None):
    custom_env = os.environ.copy()
    custom_env['TOKENIZERS_PARALLELISM'] = 'false'
    if device_id is not None:
        custom_env['CUDA_VISIBLE_DEVICES'] = device_id
    return custom_env

RADCLIQ_METRIC_NAMES = ['bleu_score', 'bertscore', 'semb_score', 'radgraph_combined', 'RadCliQ-v0', 'RadCliQ-v1']

def invoke_radcliq_process(gt_reports, gen_reports, device_id=None, remove_tmp_files=True, verbose=False):
    assert len(gt_reports) == len(gen_reports)

    timestamp = get_timestamp()

    # Define input & output paths
    input_gt_path = os.path.join(TMP_FOLDER, f'radcliq-gt_{timestamp}.csv')
    input_gen_path = os.path.join(TMP_FOLDER, f'radcliq-gen_{timestamp}.csv')
    output_path = os.path.join(TMP_FOLDER, f'radcliq-output_{timestamp}.csv')

    # Create input files
    os.makedirs(TMP_FOLDER, exist_ok=True)
    in_gt_df = pd.DataFrame(columns=['study_id', 'report'], data=[(i, r) for i, r in enumerate(gt_reports)])
    in_gt_df.to_csv(input_gt_path, index=False)
    in_gen_df = pd.DataFrame(columns=['study_id', 'report'], data=[(i, r) for i, r in enumerate(gen_reports)])
    in_gen_df.to_csv(input_gen_path, index=False)

    # Build command & call RadCliQ process
    cmd = (f'cd {RADCLIQ_FOLDER} && '
           f'conda run -n {RADCLIQ_CONDA_ENV} python run_metric.py '
           f'--gt_reports_path "{input_gt_path}" '
           f'--gen_reports_path "{input_gen_path}" '
           f'--output_path "{output_path}"')
    try:            
        print(f'Running RadCliQ over {len(in_gt_df)} reports ...')
        start = time.time()
        if verbose:
            print(f'\tCommand = {cmd}')        
        result = subprocess.run(
            cmd, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=_get_custom_env(device_id),
        )
        print(f'RadCliq process done. Elapsed seconds = {time.time() - start}')
        if verbose:
            print("stdout:", result.stdout.decode('utf-8'))
            print("stderr:", result.stderr.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print("Error running RadCliq command:")
        print("stdout:", e.stdout.decode('utf-8'))
        print("stderr:", e.stderr.decode('utf-8'))
        raise e

    # Read RadCliQ output
    out_df = pd.read_csv(output_path)

    assert len(out_df) == len(gt_reports)

    # Remove tmp files
    if remove_tmp_files:
        os.remove(input_gt_path)
        os.remove(input_gen_path)
        os.remove(output_path)

    return { metric_name: out_df[metric_name].to_numpy() for metric_name in RADCLIQ_METRIC_NAMES }