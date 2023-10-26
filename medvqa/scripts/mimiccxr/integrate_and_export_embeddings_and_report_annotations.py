import argparse
import random
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from medvqa.models.huggingface_utils import (
    CachedTextEmbeddingExtractor,
    SupportedHuggingfaceMedicalBERTModels,
)
from medvqa.utils.files import (
    get_cached_jsonl_file,
    get_file_path_with_hashing_if_too_long,
    load_json,
    save_pickle,
    get_cached_pickle_file,
)
from medvqa.datasets.mimiccxr import MIMICCXR_LARGE_FAST_CACHE_DIR
from medvqa.utils.logging import print_blue, print_bold

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--integrated_fact_metadata_filepath', type=str, required=True)
    parser.add_argument('--integrated_sentence_facts_filepath', type=str, required=True)
    parser.add_argument('--integrated_chest_imagenome_observations_filepath', type=str, required=True)
    parser.add_argument('--integrated_chest_imagenome_anatomical_locations_filepath', type=str, required=True)
    parser.add_argument('--cluster_based_labels_per_report_filepath', type=str, required=True)
    parser.add_argument('--background_findings_and_impression_per_report_filepath', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument('--device', type=str, default='GPU', choices=['CPU', 'GPU'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_checkpoint_folder_path', type=str, required=True)
    # clustering parameters
    parser.add_argument('--num_kmeans_clusters__sentences', type=int, default=400)
    parser.add_argument('--num_kmeans_clusters__facts', type=int, default=400)
    parser.add_argument('--num_kmeans_clusters__anatloc', type=int, default=300)
    parser.add_argument('--num_kmeans_iterations', type=int, default=300)

    args = parser.parse_args()

    # Load integrated sentence facts
    print_bold('Loading integrated sentence facts from:', args.integrated_sentence_facts_filepath)
    integrated_sentence_facts = get_cached_jsonl_file(args.integrated_sentence_facts_filepath)
    sentence2facts = { row['sentence']: row['facts'] for row in integrated_sentence_facts }

    # Load integrated fact metadata
    print_bold('Loading integrated fact metadata from:', args.integrated_fact_metadata_filepath)
    integrated_fact_metadata = get_cached_jsonl_file(args.integrated_fact_metadata_filepath)
    fact2metadata = { row['fact'] : row for row in integrated_fact_metadata }

    # Load cluster based labels per report
    print_bold('Loading cluster based labels per report from:', args.cluster_based_labels_per_report_filepath)
    cblpr = get_cached_pickle_file(args.cluster_based_labels_per_report_filepath)
    label_id_2_idx = {x:i for i, x in enumerate(cblpr['top_label_ids'])}
    assert len(label_id_2_idx) == len(cblpr['top_label_names'])
    cluster_based_label_names = cblpr['top_label_names']

    # Load background, findings and impression per report
    print_bold('Loading background, findings and impression per report from:', args.background_findings_and_impression_per_report_filepath)
    bfipr = load_json(args.background_findings_and_impression_per_report_filepath)

    sentences = []
    sentence2idx = {}
    facts = []
    fact2idx = {}

    reports = [{} for _ in range(len(bfipr))]
    for i, row in tqdm(enumerate(bfipr), total=len(bfipr), mininterval=2.0):
        report = reports[i]
        report_filepath = row['path']
        relative_path = report_filepath[report_filepath.index('mimic-cxr'):]
        parts = relative_path.split('/')
        report['path'] = relative_path
        report['part_id'] = parts[-3]
        report['subject_id'] = parts[-2]
        report['study_id'] = parts[-1].split('.')[0]
        with open(report_filepath, 'r') as f:
            report['original_report'] = f.read()
        report['background'] = row['background']
        report['findings'] = row['findings']
        report['impression'] = row['impression']
        report['findings_sentence_idxs'] = []
        report['findings_fact_idxs'] = []
        report['impression_sentence_idxs'] = []
        report['impression_fact_idxs'] = []
        for x, name in zip((row['findings'], row['impression']), ('findings', 'impression')):
            if x:
                for s in sent_tokenize(x):
                    try:
                        s_idx = sentence2idx[s]
                    except KeyError:
                        s_idx = len(sentences)
                        sentences.append(s)
                        sentence2idx[s] = s_idx
                    report[f'{name}_sentence_idxs'].append(s_idx)
                    for f in sentence2facts[s]:
                        try:
                            f_idx = fact2idx[f]
                        except KeyError:
                            f_idx = len(facts)
                            facts.append(f)
                            fact2idx[f] = f_idx
                        report[f'{name}_fact_idxs'].append(f_idx)
        report['cluster_based_labels'] = [label_id_2_idx[x] for x in cblpr['labeled_reports'][i]['labels']]
    
    sentence_fact_idxs = []
    for s in sentences:
        sentence_fact_idxs.append([fact2idx[f] for f in sentence2facts[s]])
    
    # Obtain embeddings for each sentence
    print_bold('Obtaining embeddings for each sentence...')
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name, device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
    )
    sentence_embeddings = embedding_extractor.compute_text_embeddings(sentences)
    sentence_cluster_ids = embedding_extractor.compute_kmeans_labels(sentences,
                                                                     num_clusters=args.num_kmeans_clusters__sentences,
                                                                     num_iterations=args.num_kmeans_iterations,
                                                                     embeddings=sentence_embeddings)
    
    # Obtain embeddings for each fact
    print_bold('Obtaining embeddings for each fact...')
    fact_embeddings = embedding_extractor.compute_text_embeddings(facts)
    fact_cluster_ids = embedding_extractor.compute_kmeans_labels(facts,
                                                                 num_clusters=args.num_kmeans_clusters__facts,
                                                                 num_iterations=args.num_kmeans_iterations,
                                                                 embeddings=fact_embeddings)
    
    # Obtain embeddings for each anatomical location
    print_bold('Obtaining embeddings for each anatomical location...')
    anatomical_locations_set = set()
    for f in facts:
        anatloc = fact2metadata[f]['metadata']['anatomical location']
        if anatloc:
            anatomical_locations_set.add(anatloc)
    anatomical_locations = list(anatomical_locations_set)
    anatomical_locations.sort()
    print('Number of unique anatomical locations:', len(anatomical_locations))
    print('Example anatomical locations:')
    for x in random.sample(anatomical_locations, 4):
        print(x)
    anatomical_location_embeddings = embedding_extractor.compute_text_embeddings(anatomical_locations)
    anatomical_location_cluster_ids = embedding_extractor.compute_kmeans_labels(anatomical_locations,
                                                                                 num_clusters=args.num_kmeans_clusters__anatloc,
                                                                                 num_iterations=args.num_kmeans_iterations,
                                                                                 embeddings=anatomical_location_embeddings)
    
    # Load integrated_chest_imagenome_observations
    print_bold('Loading integrated_chest_imagenome_observations from:', args.integrated_chest_imagenome_observations_filepath)
    integrated_chest_imagenome_observations = get_cached_pickle_file(args.integrated_chest_imagenome_observations_filepath)
    observation_names = integrated_chest_imagenome_observations['label_names']
    sentence2chimgobs = {}
    for group in integrated_chest_imagenome_observations['groups']:
        g_sentences = group['sentences']
        g_labels = group['labels']
        for i, s in enumerate(g_sentences):
            obs_list = []
            for j, l in enumerate(g_labels[i]):
                if l:
                    obs_list.append(observation_names[j])
            sentence2chimgobs[s] = obs_list

    # Load integrated_chest_imagenome_anatomical_locations
    print_bold('Loading integrated_chest_imagenome_anatomical_locations from:', args.integrated_chest_imagenome_anatomical_locations_filepath)
    integrated_chest_imagenome_anatomical_locations = get_cached_pickle_file(args.integrated_chest_imagenome_anatomical_locations_filepath)
    anatomical_location_names = integrated_chest_imagenome_anatomical_locations['label_names']
    sentence2chimganatloc = {}
    for group in integrated_chest_imagenome_anatomical_locations['groups']:
        g_sentences = group['sentences']
        g_labels = group['labels']
        for i, s in enumerate(g_sentences):
            anatloc_list = []
            for j, l in enumerate(g_labels[i]):
                if l:
                    anatloc_list.append(anatomical_location_names[j])
            sentence2chimganatloc[s] = anatloc_list

    # Integrate labels for each fact
    print_bold('Integrating labels for each fact...')
    fact_metadata = [None] * len(facts)
    for i, f in enumerate(facts):
        row = fact2metadata[f]
        metadata = row['metadata']
        anatloc = metadata['anatomical location']
        category = metadata['category']
        health_status = metadata['health status']
        comparison_status = metadata['comparison status']
        if 'improved_comparison' in row:
            comparison_status = row['improved_comparison']['comparison']
        fact_metadata[i] = {
            'anatomical_location': anatloc,
            'category': category,
            'health_status': health_status,
            'comparison_status': comparison_status or 'no comparison', # 'no comparison' is the default value
            'chest_imagenome_observations': sentence2chimgobs.get(f, []),
            'chest_imagenome_anatomical_locations': sentence2chimganatloc.get(f, []),
        }
    
    # Integrate output
    output = {
        'reports': reports,
        'cluster_based_label_names': cluster_based_label_names,
        'sentences': sentences,
        'sentence_embeddings': sentence_embeddings,
        'sentence_cluster_ids': sentence_cluster_ids,
        'sentence_fact_idxs': sentence_fact_idxs,
        'facts': facts,
        'fact_embeddings': fact_embeddings,
        'fact_cluster_ids': fact_cluster_ids,
        'fact_metadata': fact_metadata,
        'anatomical_locations': anatomical_locations,
        'anatomical_location_embeddings': anatomical_location_embeddings,
        'anatomical_location_cluster_ids': anatomical_location_cluster_ids,
    }
    save_path = get_file_path_with_hashing_if_too_long(
        folder_path=MIMICCXR_LARGE_FAST_CACHE_DIR,
        prefix='integrated_report_annotations',
        strings=[
            args.integrated_fact_metadata_filepath,
            args.integrated_sentence_facts_filepath,
            args.integrated_chest_imagenome_observations_filepath,
            args.integrated_chest_imagenome_anatomical_locations_filepath,
            args.cluster_based_labels_per_report_filepath,
            args.background_findings_and_impression_per_report_filepath,
            args.model_name,
            args.model_checkpoint_folder_path,
            args.device,
            str(args.batch_size),
            str(args.num_workers),
            str(args.num_kmeans_clusters__sentences),
            str(args.num_kmeans_clusters__facts),
            str(args.num_kmeans_clusters__anatloc),
            str(args.num_kmeans_iterations),
        ],
        force_hashing=True,
    )
    print_blue('Saving output to:', save_path, bold=True)
    save_pickle(output, save_path)
    print('Done!')

if __name__ == '__main__':
    main()