import argparse
import os
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor, SupportedHuggingfaceMedicalBERTModels
from medvqa.utils.hashing import hash_string_list
from medvqa.utils.files import load_jsonl, save_pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated_sentence_facts_filepath", type=str, required=True)
    parser.add_argument("--integrated_facts_metadata_filepath", type=str, required=True)
    parser.add_argument('--paraphrased_anatomical_locations_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--paraphrased_observations_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--hard_triplets_filepaths', type=str, nargs='+', required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--num_sentence_clusters", type=int, default=400)
    parser.add_argument("--num_observation_clusters", type=int, default=400)
    parser.add_argument("--num_anatomical_location_clusters", type=int, default=200)
    parser.add_argument("--num_iterations", type=int, default=300)
    parser.add_argument("--model_name", type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument("--model_checkpoint_folder_path", type=str, default=None)    

    args = parser.parse_args()

    sentences = set()
    observations = set()
    anatomical_locations = set()

    # Load sentences and facts
    print(f"Loading integrated sentence facts from {args.integrated_sentence_facts_filepath}")
    sentence_facts = load_jsonl(args.integrated_sentence_facts_filepath)
    for row in sentence_facts:
        sentences.add(row['sentence'])
        observations.update(row['facts'])

    # Load fact metadata
    print(f"Loading integrated facts metadata from {args.integrated_facts_metadata_filepath}")
    facts_metadata = load_jsonl(args.integrated_facts_metadata_filepath)
    for row in facts_metadata:
        fact = row['fact']
        metadata = row['metadata']
        short_obs = metadata['short observation']
        det_obs = metadata['detailed observation']
        anat_loc = metadata['anatomical location']
        for x in (fact, short_obs, det_obs):
            if x:
                observations.add(x)
        if anat_loc:
            anatomical_locations.add(anat_loc)

    # Load paraphrased anatomical locations
    for filepath in args.paraphrased_anatomical_locations_filepaths:
        print(f'Loding paraphrased anatomical locations from {filepath}')
        paraphrased_anatomical_locations = load_jsonl(filepath)
        for row in paraphrased_anatomical_locations:
            anatloc = row['metadata']['anatomical location']
            paraphrases = row['parsed_response']
            anatomical_locations.add(anatloc)
            anatomical_locations.update(paraphrases)
    
    # Load paraphrased observations
    for filepath in args.paraphrased_observations_filepaths:
        print(f'Loding paraphrased observations from {filepath}')
        paraphrased_observations = load_jsonl(filepath)
        for row in paraphrased_observations:
            try:
                observation = row['metadata']['query']
            except KeyError:
                observation = row['metadata']['observation'] # backward compatibility
            paraphrases = row['parsed_response']
            observations.add(observation)
            observations.update(paraphrases)

    # Load hard triplets
    for filepath in args.hard_triplets_filepaths:
        print(f'Loding hard triplets from {filepath}')
        hard_triplets = load_jsonl(filepath)
        for row in hard_triplets:
            s = next(iter(row['metadata'].values()))
            parsed_response = row['parsed_response']
            observations.add(s)
            observations.update(parsed_response['positives'])
            observations.update(parsed_response['negatives'])

    # Convert to lists
    sentences = list(sentences)
    sentences.sort()
    observations = list(observations)
    observations.sort()
    anatomical_locations = list(anatomical_locations)
    anatomical_locations.sort()
    print(f'Number of sentences: {len(sentences)}')
    print(f'Number of observations: {len(observations)}')
    print(f'Number of anatomical locations: {len(anatomical_locations)}')

    # Cluster sentences, observations and anatomical locations
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name,
        device=args.device,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    sentence_cluster_ids = embedding_extractor.compute_kmeans_labels(sentences,
                                              num_clusters=args.num_sentence_clusters,
                                              num_iterations=args.num_iterations)
    observation_cluster_ids = embedding_extractor.compute_kmeans_labels(observations,
                                                num_clusters=args.num_observation_clusters,
                                                num_iterations=args.num_iterations)
    anatomical_location_cluster_ids = embedding_extractor.compute_kmeans_labels(anatomical_locations,
                                                    num_clusters=args.num_anatomical_location_clusters,
                                                    num_iterations=args.num_iterations)
    
    # Integrate and save
    output = {
        'sentences': [],
        'cluster_ids': [],
    }
    offset = 0
    output['sentences'].extend(sentences)
    output['cluster_ids'].extend(sentence_cluster_ids)
    offset += max(sentence_cluster_ids) + 1
    output['sentences'].extend(observations)
    output['cluster_ids'].extend([x + offset for x in observation_cluster_ids])
    offset += max(observation_cluster_ids) + 1
    output['sentences'].extend(anatomical_locations)
    output['cluster_ids'].extend([x + offset for x in anatomical_location_cluster_ids])

    h = hash_string_list([
        args.integrated_sentence_facts_filepath,
        args.integrated_facts_metadata_filepath,
        *args.paraphrased_anatomical_locations_filepaths,
        *args.paraphrased_observations_filepaths,
        *args.hard_triplets_filepaths,
        str(args.num_sentence_clusters),
        str(args.num_observation_clusters),
        str(args.num_anatomical_location_clusters),
        str(args.num_iterations),
        args.model_name,
        args.model_checkpoint_folder_path,
    ])
    output_filepath = os.path.join(MIMICCXR_CACHE_DIR, f'sentence_cluster_ids{h}.pkl')
    print(f'Saving sentence cluster ids to {output_filepath}')
    save_pickle(output, output_filepath)
