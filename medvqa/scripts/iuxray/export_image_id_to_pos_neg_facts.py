import argparse
from tqdm import tqdm
from medvqa.models.huggingface_utils import CachedTextEmbeddingExtractor
from medvqa.utils.files import get_file_path_with_hashing_if_too_long, load_json, save_pickle
from medvqa.datasets.iuxray import IUXRAY_LARGE_FAST_CACHE_DIR, IUXRAY_REPORTS_MIN_JSON_PATH
from medvqa.utils.files import load_json, load_jsonl
from medvqa.utils.logging import print_blue

def compute_positive_negative_facts_per_image(
    manual_tags_with_slashes_to_sentences_jsol_filepath,
    fact_embedding_model_name,
    fact_embedding_model_checkpoint_folder_path,
    fact_embedding_batch_size,
    fact_embedding_num_workers,
):
    print(f'Loading IUXray reports from {IUXRAY_REPORTS_MIN_JSON_PATH}...')
    reports = load_json(IUXRAY_REPORTS_MIN_JSON_PATH)

    print(f'Loading manual tags with slashes to sentences from {manual_tags_with_slashes_to_sentences_jsol_filepath}...')
    manual_tags_with_slashes_to_sentences = load_jsonl(manual_tags_with_slashes_to_sentences_jsol_filepath)
    tag2fact = { x['metadata']['query']: x['parsed_response'] for x in manual_tags_with_slashes_to_sentences }
    print(f'Loaded {len(tag2fact)} tags.')

    facts = []
    fact2idx = {}
    report_filename_to_pos_fact_idxs = {}

    # Collect positive facts per report
    for report in tqdm(reports.values(), total=len(reports), mininterval=2):
        report_filename = report['filename']
        tags_manual = report['tags_manual']
        tags_auto = report['tags_auto']
        tags_manual = [tag2fact.get(tag, tag) for tag in tags_manual]
        assert all('/' not in tag for tag in tags_manual) # all tags should be properly parsed
        assert all('/' not in tag for tag in tags_auto) # all tags should be properly parsed
        tags = tags_manual + tags_auto
        pos_fact_idxs = []
        for tag in tags:
            if tag not in fact2idx:
                fact2idx[tag] = len(facts)
                facts.append(tag)
            pos_fact_idxs.append(fact2idx[tag])
        pos_fact_idxs = list(set(pos_fact_idxs))
        pos_fact_idxs.sort()
        assert len(pos_fact_idxs) > 0
        report_filename_to_pos_fact_idxs[report_filename] = pos_fact_idxs

    # Collect negative facts per report
    print('Computing negative facts...')
    report_filename_to_neg_fact_idxs = {}
    for report in tqdm(reports.values(), total=len(reports), mininterval=2):
        report_filename = report['filename']
        pos_fact_idxs = report_filename_to_pos_fact_idxs[report_filename]
        pos_fact_idxs_set = set(pos_fact_idxs)
        neg_fact_idxs = [i for i in range(len(facts)) if i not in pos_fact_idxs_set]
        report_filename_to_neg_fact_idxs[report_filename] = neg_fact_idxs

    # Assign positive and negative facts to each image
    print('Assigning positive and negative facts to each image...')
    image_id_to_pos_neg_facts = {}
    for report in tqdm(reports.values(), total=len(reports), mininterval=2):
        report_filename = report['filename']
        images = report['images']
        image_ids = [image['id'] for image in images]
        pos_fact_idxs = report_filename_to_pos_fact_idxs[report_filename]
        neg_fact_idxs = report_filename_to_neg_fact_idxs[report_filename]
        for image_id in image_ids:
            image_id_to_pos_neg_facts[image_id] = (pos_fact_idxs, neg_fact_idxs)

    # Compute embeddings
    fact_encoder = CachedTextEmbeddingExtractor(
        model_name=fact_embedding_model_name,
        model_checkpoint_folder_path=fact_embedding_model_checkpoint_folder_path,
        batch_size=fact_embedding_batch_size,
        num_workers=fact_embedding_num_workers,
        device='cuda',
    )
    fact_embeddings = fact_encoder.compute_text_embeddings(facts)

    # Save output
    filepath_strings = [
        manual_tags_with_slashes_to_sentences_jsol_filepath,
        fact_embedding_model_name,
        fact_embedding_model_checkpoint_folder_path,
    ]
    output = {
        'facts': facts,
        'embeddings': fact_embeddings,
        'image_id_to_pos_neg_facts': image_id_to_pos_neg_facts,
    }
    output_filepath = get_file_path_with_hashing_if_too_long(
        folder_path=IUXRAY_LARGE_FAST_CACHE_DIR,
        prefix='image_id_to_pos_neg_facts',
        strings=filepath_strings,
        force_hashing=True,
    )
    print(f'Saving {output_filepath}...')
    save_pickle(output, output_filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manual_tags_with_slashes_to_sentences_jsol_filepath', type=str, required=True)
    parser.add_argument('--fact_embedding_model_name', type=str, required=True)
    parser.add_argument('--fact_embedding_model_checkpoint_folder_path', type=str, required=True)
    parser.add_argument('--fact_embedding_batch_size', type=int, default=32)
    parser.add_argument('--fact_embedding_num_workers', type=int, default=4)
    args = parser.parse_args()

    print_blue(f'Computing positive and negative facts per image...')
    compute_positive_negative_facts_per_image(
        manual_tags_with_slashes_to_sentences_jsol_filepath=args.manual_tags_with_slashes_to_sentences_jsol_filepath,
        fact_embedding_model_name=args.fact_embedding_model_name,
        fact_embedding_model_checkpoint_folder_path=args.fact_embedding_model_checkpoint_folder_path,
        fact_embedding_batch_size=args.fact_embedding_batch_size,
        fact_embedding_num_workers=args.fact_embedding_num_workers,
    )
    print_blue(f'Done!', bold=True)

if __name__ == '__main__':
    main()