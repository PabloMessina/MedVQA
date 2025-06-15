import argparse
import os
import logging
from medvqa.datasets.mimiccxr import MIMICCXR_CACHE_DIR
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor, SupportedHuggingfaceMedicalBERTModels
from medvqa.scripts.chest_imagenome.generate_phrase_groundings_from_scene_graphs import collect_phrase_groundings_from_padchest_gr_and_mscxr
from medvqa.utils.hashing_utils import hash_string_list
from medvqa.utils.files_utils import load_jsonl, load_pickle, save_pickle
from medvqa.utils.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated_sentence_facts_filepath", type=str, required=True)
    parser.add_argument("--integrated_facts_metadata_filepath", type=str, required=True)
    parser.add_argument('--paraphrased_anatomical_locations_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--paraphrased_observations_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--hard_triplets_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--revised_groundings_file', type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"])
    parser.add_argument("--num_clusters", type=int, default=100)
    parser.add_argument("--num_iterations", type=int, default=300)
    parser.add_argument("--model_name", type=str, required=True, choices=SupportedHuggingfaceMedicalBERTModels.get_all())
    parser.add_argument("--model_checkpoint_folder_path", type=str, default=None)    

    args = parser.parse_args()

    sentences = set()

    # Load sentences and facts
    logger.info(f"Loading integrated sentence facts from {args.integrated_sentence_facts_filepath}")
    sentence_facts = load_jsonl(args.integrated_sentence_facts_filepath)
    for row in sentence_facts:
        sentences.update(row['facts'])

    # Load fact metadata
    logger.info(f"Loading integrated facts metadata from {args.integrated_facts_metadata_filepath}")
    facts_metadata = load_jsonl(args.integrated_facts_metadata_filepath)
    for row in facts_metadata:
        fact = row['fact']
        metadata = row['metadata']
        short_obs = metadata['short observation']
        det_obs = metadata['detailed observation']
        anat_loc = metadata['anatomical location']
        for x in (fact, short_obs, det_obs):
            if x:
                sentences.add(x)
        if anat_loc:
            sentences.add(anat_loc)

    # Load paraphrased anatomical locations
    for filepath in args.paraphrased_anatomical_locations_filepaths:
        logger.info(f'Loding paraphrased anatomical locations from {filepath}')
        paraphrased_anatomical_locations = load_jsonl(filepath)
        for row in paraphrased_anatomical_locations:
            anatloc = row['metadata']['anatomical location']
            paraphrases = row['parsed_response']
            sentences.add(anatloc)
            sentences.update(paraphrases)
    
    # Load paraphrased observations
    for filepath in args.paraphrased_observations_filepaths:
        logger.info(f'Loding paraphrased observations from {filepath}')
        paraphrased_observations = load_jsonl(filepath)
        for row in paraphrased_observations:
            try:
                observation = row['metadata']['query']
            except KeyError:
                observation = row['metadata']['observation'] # backward compatibility
            paraphrases = row['parsed_response']
            sentences.add(observation)
            sentences.update(paraphrases)

    # Load hard triplets
    for filepath in args.hard_triplets_filepaths:
        logger.info(f'Loding hard triplets from {filepath}')
        hard_triplets = load_jsonl(filepath)
        for row in hard_triplets:
            s = next(iter(row['metadata'].values()))
            parsed_response = row['parsed_response']
            sentences.add(s)
            sentences.update(parsed_response['positives'])
            sentences.update(parsed_response['negatives'])

    # Load revised groundings
    logger.info(f'Loading revised groundings from {args.revised_groundings_file}')
    revised_data = load_pickle(args.revised_groundings_file)
    logger.info(f"Loaded {len(revised_data)} entries from {args.revised_groundings_file}")

    collected_groundings = collect_phrase_groundings_from_padchest_gr_and_mscxr()
    logger.info(f"Collected {len(collected_groundings)} groundings from PadChest-GR and MS-CXR")
    
    sentences.update(phrase for entry in revised_data for phrase in entry["phrase2locations"].keys())
    sentences.update(item['phrase'] for item in collected_groundings)

    # Convert to lists
    sentences = list(sentences)
    sentences.sort()
    logger.info(f'Number of sentences: {len(sentences)}')

    # Cluster sentences, observations and anatomical locations
    embedding_extractor = CachedTextEmbeddingExtractor(
        model_name=args.model_name,
        device=args.device,
        model_checkpoint_folder_path=args.model_checkpoint_folder_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    sentence_cluster_ids = embedding_extractor.compute_kmeans_labels(sentences,
                                              num_clusters=args.num_clusters,
                                              num_iterations=args.num_iterations)
    
    # Integrate and save
    output = {
        'sentences': sentences,
        'cluster_ids': sentence_cluster_ids,
    }

    h = hash_string_list([
        args.integrated_sentence_facts_filepath,
        args.integrated_facts_metadata_filepath,
        *args.paraphrased_anatomical_locations_filepaths,
        *args.paraphrased_observations_filepaths,
        *args.hard_triplets_filepaths,
        str(args.num_clusters),
        str(args.num_iterations),
        args.model_name,
        str(args.model_checkpoint_folder_path),
        args.revised_groundings_file
    ])
    output_filepath = os.path.join(MIMICCXR_CACHE_DIR, f'sentence_cluster_ids{h}.pkl')
    logger.info(f'Saving sentence cluster ids to {output_filepath}')
    save_pickle(output, output_filepath)
