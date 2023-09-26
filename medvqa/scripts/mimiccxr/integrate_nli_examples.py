import argparse
import os
from medvqa.datasets.mimiccxr import MIMICCXR_FAST_CACHE_DIR
from medvqa.utils.files import save_jsonl, load_jsonl
from medvqa.datasets.nli import (
    MEDNLI_DEV_JSONL_PATH,
    MEDNLI_TEST_JSONL_PATH,
    MEDNLI_TRAIN_JSONL_PATH,
    RADNLI_DEV_JSONL_PATH,
)

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_nli_examples_jsonl_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--generated_nli_examples_v2_jsonl_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--generated_nli_examples_v3_jsonl_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--generated_nli_examples_v5_jsonl_filepaths', type=str, nargs='+', required=True)
    parser.add_argument('--generation_methods', type=str, nargs='+', required=True)
    args = parser.parse_args()
    n1 = len(args.generated_nli_examples_jsonl_filepaths)
    n2 = len(args.generated_nli_examples_v2_jsonl_filepaths)
    n3 = len(args.generated_nli_examples_v3_jsonl_filepaths)
    n5 = len(args.generated_nli_examples_v5_jsonl_filepaths)
    assert n1 + n2 + n3 + n5 == len(args.generation_methods), 'Number of generation methods must match number of files'
    generation_methods_1 = args.generation_methods[:n1]
    generation_methods_2 = args.generation_methods[n1:n1+n2]
    generation_methods_3 = args.generation_methods[n1+n2:n1+n2+n3]
    generation_methods_5 = args.generation_methods[n1+n2+n3:n1+n2+n3+n5]

    output = []

    # Load generated NLI examples
    assert len(args.generated_nli_examples_jsonl_filepaths) == len(generation_methods_1)
    for filepath, source in zip(args.generated_nli_examples_jsonl_filepaths, generation_methods_1):
        assert os.path.exists(filepath), f'File {filepath} does not exist'
        rows = load_jsonl(filepath)
        print(f'Loaded {len(rows)} rows from {filepath}')
        for row in rows:
            premise = row['metadata']['query']
            entailment = row['parsed_response']['e']
            contradiction = row['parsed_response']['c']
            neutral = row['parsed_response']['n']
            # entailment
            output.append({
                'premise': premise,
                'hypothesis': entailment,
                'label': 'entailment',
                'source': source,
            })
            # neutral
            output.append({
                'premise': premise,
                'hypothesis': neutral,
                'label': 'neutral',
                'source': source,
            })
            # contradiction works both ways
            output.append({
                'premise': premise,
                'hypothesis': contradiction,
                'label': 'contradiction',
                'source': source,
            })
            output.append({
                'premise': contradiction,
                'hypothesis': premise,
                'label': 'contradiction',
                'source': source,
            })

    # Load generated NLI examples v2
    assert len(args.generated_nli_examples_v2_jsonl_filepaths) == len(generation_methods_2)
    for filepath, source in zip(args.generated_nli_examples_v2_jsonl_filepaths, generation_methods_2):
        assert os.path.exists(filepath), f'File {filepath} does not exist'
        rows = load_jsonl(filepath)
        print(f'Loaded {len(rows)} rows from {filepath}')
        for row in rows:
            examples = row['parsed_response']
            for example in examples:
                output.append({
                    'premise': example['P'],
                    'hypothesis': example['H'],
                    'label': example['L'],
                    'source': source,
                })
                if example['L'] == 'contradiction': # contradiction works both ways
                    output.append({
                        'premise': example['H'],
                        'hypothesis': example['P'],
                        'label': example['L'],
                        'source': source,
                    })

    # Load generated NLI examples v3
    assert len(args.generated_nli_examples_v3_jsonl_filepaths) == len(generation_methods_3)
    for filepath, source in zip(args.generated_nli_examples_v3_jsonl_filepaths, generation_methods_3):
        assert os.path.exists(filepath), f'File {filepath} does not exist'
        rows = load_jsonl(filepath)
        print(f'Loaded {len(rows)} rows from {filepath}')
        count = 0
        for row in rows:
            query = row['metadata']['query']
            if query.startswith('P: '):
                query = query.split('\n')
                assert len(query) == 2
                assert query[0].startswith('P: ')
                assert query[1].startswith('H: ')
                premise = query[0][3:].strip()
                hypothesis = query[1][3:].strip()
            else:
                assert query.startswith('Premise: ')
                query = query.split(' | Hypothesis: ')
                assert len(query) == 2
                premise = query[0][9:].strip()
                hypothesis = query[1].strip()
            label = row['parsed_response']
            assert label in ['entailment', 'neutral', 'contradiction']
            output.append({
                'premise': premise,
                'hypothesis': hypothesis,
                'label': label,
                'source': source,
            })
            if label == 'contradiction':
                output.append({
                    'premise': hypothesis,
                    'hypothesis': premise,
                    'label': label,
                    'source': source,
                })
            count += 1
            if count < 2:
                print(output[-1])

    # Load generated NLI examples v5
    assert len(args.generated_nli_examples_v5_jsonl_filepaths) == len(generation_methods_5)
    for filepath, source in zip(args.generated_nli_examples_v5_jsonl_filepaths, generation_methods_5):
        assert os.path.exists(filepath), f'File {filepath} does not exist'
        rows = load_jsonl(filepath)
        print(f'Loaded {len(rows)} rows from {filepath}')
        for row in rows:
            premise = row['metadata']['query']
            contradictions = row['parsed_response']
            for contradiction in contradictions:
                output.append({
                    'premise': premise,
                    'hypothesis': contradiction,
                    'label': 'contradiction',
                    'source': source,
                })
                output.append({ # contradiction works both ways
                    'premise': contradiction,
                    'hypothesis': premise,
                    'label': 'contradiction',
                    'source': source,
                })

    # Load examples from MedNLI and RadNLI
    for filepath, source in zip(
        [MEDNLI_TRAIN_JSONL_PATH, MEDNLI_DEV_JSONL_PATH, MEDNLI_TEST_JSONL_PATH, RADNLI_DEV_JSONL_PATH],
        ['mednli_train', 'mednli_dev', 'mednli_test', 'radnli_dev'],
    ):
        assert os.path.exists(filepath), f'File {filepath} does not exist'
        rows = load_jsonl(filepath)
        print(f'Loaded {len(rows)} rows from {filepath}')
        for row in rows:
            output.append({
                'premise': row['sentence1'],
                'hypothesis': row['sentence2'],
                'label': row['gold_label'],
                'source': source,
            })

    # Remove duplicates
    output = list({(row['premise'], row['hypothesis'], row['label'], row['source']): row for row in output}.values())
    print(f'Integrated {len(output)} NLI examples')

    # Save integrated NLI examples
    sentence_total_length = 0
    facts_total_length = 0
    for row in output:
        sentence_total_length += len(row['premise'])
        sentence_total_length += len(row['hypothesis'])
    integrated_nli_examples_filepath = os.path.join(MIMICCXR_FAST_CACHE_DIR,
                                                    f'integrated_nli_examples({len(output)},'
                                                    f'{sentence_total_length}).jsonl')
    print(f'Saving integrated NLI examples to {integrated_nli_examples_filepath}')
    save_jsonl(output, integrated_nli_examples_filepath)
    print('Done!')