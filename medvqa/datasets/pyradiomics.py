import os
import time
import subprocess
import pandas as pd
import numpy as np
from skimage import io
from skimage.color import rgb2gray
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor
from medvqa.utils.files import load_pickle, save_to_pickle

class PyradiomicsFeatureExtractor:
    
    def __init__(self):
        radiomics.setVerbosity(40)
        extractor = featureextractor.RadiomicsFeatureExtractor()
        extractor.enableAllFeatures()
        extractor.enableFeatureClassByName('shape', False)
        extractor.enableImageTypes(Original={}, Wavelet={}, LBP2D={}, Square={}, Logarithm={})
        self.extractor = extractor

    def __call__(self, image_path):
        original = io.imread(image_path)
        grayscale = rgb2gray(original)
        img = sitk.GetImageFromArray(grayscale)
        mask = np.ones(grayscale.shape, dtype='uint8')
        mask[0][0] = 0
        mask = sitk.GetImageFromArray(mask)    
        features = self.extractor.execute(img, mask)
        feat_flat = []
        for key in features.keys():
            if key.startswith('diag'):
                continue
            val = features[key]
            if type(val) is np.float64:
                feat_flat.append(val)
            else:
                assert type(val) is np.ndarray
                if len(val.shape) == 0:
                    feat_flat.append(val + 0)
                else:
                    feat_flat.extend(val)
        feat_flat = np.array(feat_flat, dtype=np.float64)
        return feat_flat

    def process_batch(self, image_paths, verbose=True, n_prints=40, batch_id=None):
        n = len(image_paths)
        assert n > 0
        if verbose:
            print(f'Extracting pyradiomics features for {n} images ...')
        start = time.time()
        feat_matrix = None
        k = n // n_prints
        for i, image_path in enumerate(image_paths):
            if verbose and (i+1) % k == 0:
                if batch_id is not None:
                    print(f'id={batch_id}, {i+1}/{n}, elapsed_time={(time.time() - start):.2f}')
                else:
                    print(f'{i+1}/{n}, elapsed_time={(time.time() - start):.2f}')
            feat = self.__call__(image_path)
            if feat_matrix is None:
                feat_matrix = np.empty((n, len(feat)), dtype=float)
            feat_matrix[i] = feat
        assert feat_matrix is not None
        print('feat_matrix.shape =', feat_matrix.shape)
        return feat_matrix

class ChunkJob:
    def __init__(self, tmp_folder, script_path, chunk_id, image_paths, input_filename, output_filename):
        
        self.image_paths = image_paths
        self.input_path = os.path.join(tmp_folder, input_filename)
        self.output_path = os.path.join(tmp_folder, output_filename)

        # Create input file        
        in_df = pd.DataFrame(image_paths)
        in_df.to_csv(self.input_path, header=False, index=False)

        # Build command
        self.cmd = (f'python {script_path} --chunk-mode'
        f' --chunk-id {chunk_id} --input-path {self.input_path} --output-path {self.output_path}')

def split_work_into_chunks(tmp_folder, script_path, n_chunks, image_paths):
    n = len(image_paths)
    print(f'Splitting {n} images into {n_chunks} chunks')
    chunk_size = n // n_chunks
    r = n % n_chunks
    offset = 0
    chunk_jobs = []
    for i in range(r):
        chunk_jobs.append(ChunkJob(
            tmp_folder=tmp_folder,
            script_path=script_path,
            chunk_id=i,
            image_paths=image_paths[offset : offset+chunk_size+1],
            input_filename=f'chunk_{i}_input.csv',
            output_filename=f'chunk_{i}_output.pkl',
        ))
        actual_size = len(chunk_jobs[-1].image_paths)
        assert actual_size == chunk_size + 1
        offset += actual_size
    for i in range(r, min(n, n_chunks)):
        chunk_jobs.append(ChunkJob(
            tmp_folder=tmp_folder,
            script_path=script_path,
            chunk_id=i,
            image_paths=image_paths[offset : offset+chunk_size],
            input_filename=f'chunk_{i}_input.csv',
            output_filename=f'chunk_{i}_output.pkl',
        ))
        actual_size = len(chunk_jobs[-1].image_paths)
        assert actual_size == chunk_size
        offset += actual_size
    assert offset == n
    return chunk_jobs

def extract_features_chunk(chunk_id, input_path, output_path):
    print(f'extract_features_chunk():\n  input_path={input_path}\n  output_path={output_path}')
    df = pd.read_csv(input_path, header=None)    
    image_paths = df[0]
    extractor = PyradiomicsFeatureExtractor()
    features = extractor.process_batch(image_paths, batch_id=chunk_id)
    output = {
        'image_paths': image_paths,
        'features': features,
    }
    save_to_pickle(output, output_path)
    print(f'Features saved to {output_path}')

def extract_features_all(image_paths, script_path, tmp_folder, n_chunks, output_path):

    os.makedirs(tmp_folder, exist_ok=True)
    jobs = split_work_into_chunks(tmp_folder, script_path, n_chunks, image_paths)
    start = time.time()    

    # Spawn subprocesses
    processes = []
    for i, job in enumerate(jobs):
        print(f'#### starting process {i}')
        print(f'Command = {jobs[i].cmd}')
        processes.append(subprocess.Popen(job.cmd, shell=True))
            
    # Wait for completion
    for i, p in enumerate(processes):
        p.wait()
        print(f'**** process {i} finished, elapsed time = {time.time() - start}')
    
    # Merge chunks
    print('Merging chunks ...')
    chunks = [load_pickle(job.output_path) for job in jobs]
    
    for chunk in chunks:
        assert len(chunk['image_paths']) == len(chunk['features'])
    
    n = sum(len(chunk['image_paths']) for chunk in chunks)
    image_paths = [None] * n
    features = np.empty((n, chunks[0]['features'].shape[1]), dtype=float)
    print('len(image_paths) =', len(image_paths))
    print('features.shape =', features.shape)
    offset = 0
    for chunk in chunks:
        chunk_image_paths = chunk['image_paths']
        chunk_features = chunk['features']
        chunk_size = len(chunk_image_paths)
        for i in range(chunk_size):
            image_paths[offset + i] = chunk_image_paths[i]
            features[offset + i] = chunk_features[i]
        offset += chunk_size    
    assert offset == n

    output = {
        'image_paths': image_paths,
        'features': features,
    }
    
    save_to_pickle(output, output_path)
    print(f'Merged features saved to {output_path}')