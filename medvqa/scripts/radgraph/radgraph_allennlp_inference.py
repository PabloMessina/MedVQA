# This script has been adapted from RadGraph's inference.py (see https://physionet.org/content/radgraph/1.0.0/)
# This is intended to be run as a script from the command line or as a subprocess, e.g.:

# subprocess.run("conda run -n dygiepp python3 {path_to_this_inference_script} \
# --model_path {path_to_mode_checkpoint_folder/model.tar.gz} \
# --data_path {path_to_folder_with_txt_files} \
# --out_path {path_to_jsonl_file_to_save_the_output} \
# --temp_folder {path_to_folder_where_temporary_files_will_be_saved}", shell=True)

# The original instructions from RadGraph are copied below:

# ====================================================================================================

# Instructions for using the checkpoint for inference:

# Basic Setup (One time activity)

# 1. Clone the DYGIE++ repository from: https://github.com/dwadden/dygiepp. This repositiory is managed by Wadden et al., authors of the paper Entity, Relation, and Event Extraction with Contextualized Span Representations (https://www.aclweb.org/anthology/D19-1585.pdf).

# git clone https://github.com/dwadden/dygiepp.git

# 2. Navigate to the root of repo in your system and use the following commands to setup the conda environment:

# conda create --name dygiepp python=3.7
# pip install -r requirements.txt
# conda develop .   # Adds DyGIE to your PYTHONPATH

# Running Inference on Radiology Reports

# 3. Activate the conda environment:

# conda activate dygiepp

# 3. Copy the inference.py file to the root of the cloned repo where you have the dygie folder

# 4. Run the inference.py file using the command:

# python3 inference.py --model_path <path to file in model_checkpoint ending in tar.gz> --data_path <path to folder with reports> --out_path <path to file where to save result ending in .json> --cuda_device <optional id>

# python3 inference.py --model_path /mnt/data/radgraph/data/models/model_checkpoint/model.tar.gz --data_path /mnt/data/radgraph/data/fake_reports/ --out_path /mnt/data/radgraph/data/fake_reports/results.json

import os
import glob
import json 
import re
import argparse

def get_file_list(path, temp_folder):
    
    """Gets path to all the reports (.txt format files) in the specified folder, and
    saves it in a temporary json file
    
        Args:
            path: Path to the folder containing the reports
    """
    
    file_list = [item for item in glob.glob(f"{path}/*.txt")]
    
    # Number of files for inference at once depends on the memory available.
    ## Recemmended to use no more than batches of 25,000 files
    
    with open(os.path.join(temp_folder, 'temp_file_list.json'), 'w') as f:
        json.dump(file_list, f)

def preprocess_reports(temp_folder):
    
    """ Load up the files mentioned in the temporary json file, and
    processes them in format that the dygie model can take as input.
    Also save the processed file in a temporary file.
    """
    
    with open(os.path.join(temp_folder, 'temp_file_list.json'), 'r') as f:
        file_list = json.load(f)
    
    final_list = []
    for idx, file in enumerate(file_list):

        with open(file, 'r') as f:
            temp_file = f.read()
        
        sen = re.sub('(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )', r' ',temp_file).split()
        temp_dict = {}

        temp_dict["doc_key"] = file
        
        ## Current way of inference takes in the whole report as 1 sentence
        temp_dict["sentences"] = [sen]

        final_list.append(temp_dict)

        if(idx % 1000 == 0):
            print(f"{idx+1} reports done")
    
    print(f"{idx+1} reports done")
    
    with open(os.path.join(temp_folder, 'temp_dygie_input.json'), 'w') as outfile:
        for item in final_list:
            json.dump(item, outfile)
            outfile.write("\n")

def run_inference(model_path, dygie_package_parent_folder, temp_folder, cuda):
    
    """ Runs the inference on the processed input files. Saves the result in a
    temporary output file
    
    Args:
        model_path: Path to the model checkpoint
        cuda: GPU id
    
    
    """
    out_path = os.path.join(temp_folder, 'temp_dygie_output.json')
    data_path = os.path.join(temp_folder, 'temp_dygie_input.json')
    
    # dygie_package_parent_folder = os.path.dirname(os.path.realpath(__file__))
    print(f"dygie_package_parent_folder: {dygie_package_parent_folder}")
    assert os.path.exists(os.path.join(dygie_package_parent_folder, 'dygie'))

    # Add dygie_package_folder to the Python path so that we can import it.
    import sys
    sys.path.append(dygie_package_parent_folder)
    # Import 'dygie' to make sure it's accessible
    try:
        import dygie
        print("Imported dygie successfully")
        print(dygie.__file__)
    except ImportError:
        print("Error: 'dygie' module not found.")
        sys.exit(1)

    command = f"PYTHONPATH={dygie_package_parent_folder}:$PYTHONPATH conda run -n dygiepp allennlp predict {model_path} {data_path} \
            --predictor dygie \
            --include-package dygie \
            --use-dataset-reader \
            --output-file {out_path} \
            --cuda-device {cuda} \
            --silent"
    
    print('Running command: ', command)

    ret = os.system(command)

    if ret != 0:
        raise Exception("Command failed")
    
    assert os.path.exists(out_path)

    print("Done with inference")
    
    

def postprocess_reports(temp_folder):
    
    """Post processes all the reports and saves the result in train.json format
    """
    final_dict = {}

    file_path = os.path.join(temp_folder, 'temp_dygie_output.json')
    data = []

    with open(file_path,'r') as f:
        for line in f:
            data.append(json.loads(line))

    for file in data:
        postprocess_individual_report(file, final_dict)
    
    return final_dict

def postprocess_individual_report(file, final_dict, data_source=None):
    
    """Postprocesses individual report
    
    Args:
        file: output dict for individual reports
        final_dict: Dict for storing all the reports
    """
    
    try:
        temp_dict = {}

        temp_dict['text'] = " ".join(file['sentences'][0])
        n = file['predicted_ner'][0]
        r = file['predicted_relations'][0]
        s = file['sentences'][0]
        temp_dict["entities"] = get_entity(n,r,s)
        temp_dict["data_source"] = data_source
        temp_dict["data_split"] = "inference"

        final_dict[file['doc_key']] = temp_dict
    
    except:
        print(f"Error in doc key: {file['doc_key']}. Skipping inference on this file")
        
def get_entity(n,r,s):
    
    """Gets the entities for individual reports
    
    Args:
        n: list of entities in the report
        r: list of relations in the report
        s: list containing tokens of the sentence
        
    Returns:
        dict_entity: Dictionary containing the entites in the format similar to train.json 
    
    """

    dict_entity = {}
    rel_list = [item[0:2] for item in r]
    ner_list = [item[0:2] for item in n]
    for idx, item in enumerate(n):
        temp_dict = {}
        start_idx, end_idx, label = item[0], item[1], item[2]
        temp_dict['tokens'] = " ".join(s[start_idx:end_idx+1])
        temp_dict['label'] = label
        temp_dict['start_ix'] = start_idx
        temp_dict['end_ix'] = end_idx
        rel = []
        relation_idx = [i for i,val in enumerate(rel_list) if val== [start_idx, end_idx]]
        for i,val in enumerate(relation_idx):
            obj = r[val][2:4]
            lab = r[val][4]
            try:
                object_idx = ner_list.index(obj) + 1
            except:
                continue
            rel.append([lab,str(object_idx)])
        temp_dict['relations'] = rel
        dict_entity[str(idx+1)] = temp_dict
    
    return dict_entity

def cleanup(temp_folder):
    """Removes all the temporary files created during the inference process
    
    """
    # os.system("rm temp_file_list.json")
    # os.system("rm temp_dygie_input.json")
    # os.system("rm temp_dygie_output.json")
    file_paths_to_remove = [
        os.path.join(temp_folder, 'temp_file_list.json'),
        os.path.join(temp_folder, 'temp_dygie_input.json'),
        os.path.join(temp_folder, 'temp_dygie_output.json')
    ]
    for file_path in file_paths_to_remove:
        assert os.path.exists(file_path)
        os.remove(file_path)

def run(model_path, dygie_package_parent_folder, data_path, out_path, temp_folder, cuda):

    os.makedirs(temp_folder, exist_ok=True) # Create temp folder if it doesn't exist
    
    print("Getting paths to all the reports...")
    get_file_list(data_path, temp_folder)
    print(f"Got all the paths.")
    
    print("Preprocessing all the reports...")
    preprocess_reports(temp_folder)
    print("Done with preprocessing.")
    
    print("Running the inference now... This can take a bit of time")
    run_inference(model_path, dygie_package_parent_folder, temp_folder, cuda)
    print("Inference completed.")
    
    print("Postprocessing output file...")
    final_dict = postprocess_reports(temp_folder)
    print("Done postprocessing.")
    
    print("Saving results and performing final cleanup...")
    cleanup(temp_folder)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True) # Create out_path folder if it doesn't exist
    with open(out_path, 'w') as outfile:
        json.dump(final_dict, outfile)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, nargs='?', required=True,
                        help='path to model checkpoint')

    parser.add_argument('--dygie_package_parent_folder', type=str, nargs='?', required=True,
                        help='path to dygie package parent folder')
    
    parser.add_argument('--data_path', type=str, nargs='?', required=True,
                        help='path to folder containing reports')
    
    parser.add_argument('--out_path', type=str, nargs='?', required=True,
                        help='path to file to write results')
    
    parser.add_argument('--temp_folder', type=str, nargs='?', required=False,
                        default = "./temp", help='path to folder to store temporary files')
    
    parser.add_argument('--cuda_device', type=int, nargs='?', required=False,
                        default = -1, help='id of GPU, if to use')
    
    args = parser.parse_args()
    
    run(args.model_path, args.dygie_package_parent_folder, args.data_path, args.out_path, args.temp_folder, args.cuda_device)
