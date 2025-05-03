import argparse
from medvqa.scripts.mimiccxr.extract_comparisons_from_facts_with_openai import _ALLOWED_CATEGORIES
from medvqa.utils.files_utils import load_jsonl, save_jsonl
from collections import Counter

from medvqa.utils.logging_utils import print_orange

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--integrated_fact_metadata_filepath", type=str, required=True)
    parser.add_argument("--extracted_comparisons_filepaths", nargs="+", type=str, required=True)
    parser.add_argument('--extraction_methods', type=str, nargs='+', required=True)
    args = parser.parse_args()

    assert len(args.extracted_comparisons_filepaths) == len(args.extraction_methods)

    # Load extracted comparisons
    sent2comp = {}
    sent2extm = {}
    for fp, em in zip(args.extracted_comparisons_filepaths, args.extraction_methods):
        print(f'Loading extracted comparisons from {fp}...')
        rows = load_jsonl(fp)
        for row in rows:
            if 'metadata' in row:
                s = row['metadata']['sentence']
                c = row['parsed_response']
            else:
                s = row['sentence']
                c = row['comparison']
            assert s not in sent2comp
            sent2comp[s] = c
            sent2extm[s] = em

    # Load extracted facts and metadata
    print('Loading extracted facts and metadata...')
    integrated_fact_metadata = load_jsonl(args.integrated_fact_metadata_filepath)

    # Update extracted facts and metadata with comparisons
    new_comp_len_sum = 0
    count = 0
    comps = []
    for row in integrated_fact_metadata:
        fact = row['fact']
        metadata = row['metadata']
        comp = metadata['comparison status']
        psc = metadata['prev_study_comparison?']
        em = row['extraction_method']
        is_psc_invalid = psc not in ('yes', 'no')
        is_comp_inconsistent = (psc == 'yes') != (comp != '')
        skip = True
        if is_psc_invalid or is_comp_inconsistent:
            skip = False
        if em == 't5-small-finetuned':
            skip = False
        else:
            assert 'gpt' in em
            if (comp != '' and comp not in _ALLOWED_CATEGORIES):
                skip = False
        if skip:
            continue
        new_comp = sent2comp[fact]
        # assert new_comp in _ALLOWED_CATEGORIES, (row, new_comp, sent2extm[fact])
        if new_comp not in _ALLOWED_CATEGORIES:
            print_orange(f'Warning: {new_comp} not in allowed categories (fact: {fact}, extraction method: {sent2extm[fact]})')
            new_comp = 'other'
        row['improved_comparison'] = {
            'comparison': new_comp,
            'extraction_method': sent2extm[fact]
        }
        comps.append(new_comp)
        new_comp_len_sum += len(new_comp)
        count += 1

    # Print stats
    print(f'Updated {count}/{len(integrated_fact_metadata)} facts with comparisons.')
    counter = Counter(comps)
    for k, v in sorted(list(counter.items()), key=lambda x: x[1], reverse=True):
        print(f'{k}: {v}')

    # Save updated extracted facts and metadata
    print('Saving updated extracted facts and metadata...')
    assert args.integrated_fact_metadata_filepath.endswith('.jsonl')
    save_path = args.integrated_fact_metadata_filepath[:-len('.jsonl')] + f'.improved_comparison({new_comp_len_sum}).jsonl'
    save_jsonl(integrated_fact_metadata, save_path)
    print(f'Saved updated extracted facts and metadata to {save_path}')

    print('Done!')