import argparse
import numpy as np
from medvqa.datasets.seq2seq.seq2seq_dataset_management import (
    Seq2SeqTaskNames,
    load_gpt4_nli_examples_filepaths,
    load_ms_cxr_t_temporal_sentence_similarity_v1_data,
    load_radnli_test_data,
    load_radnli_dev_data,
)
from medvqa.models.seq2seq_utils import apply_seq2seq_model_to_sentences
from medvqa.utils.common import parsed_args_to_dict
from medvqa.utils.logging import get_console_logger, print_blue, print_bold

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint_folder_path',
        type=str,
        required=True,
        help='Path to the folder containing the checkpoint files.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use for training. Default: cuda',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size. Default: 32',
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0,
        help='Number of workers for data loading. Default: 0',
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Maximum length of the output sequence. Default: 128',
    )
    parser.add_argument(
        '--num_beams',
        type=int,
        default=1,
        help='Number of beams for beam search. Default: 1',
    )
    parser.add_argument(
        '--logging_level',
        type=str,
        default='INFO',
        help='Logging level. Default: info',
    )
    parser.add_argument(
        '--task_name',
        type=str,
        required=True,
        choices=[Seq2SeqTaskNames.NLI],
        help=f'Task name. Choices: [{Seq2SeqTaskNames.NLI}]',
    )

    parser.add_argument(
        '--gpt4_nli_examples_filepaths',
        type=str,
        nargs='+',
        default=None,
        help='Paths to the GPT-4 NLI examples files.',
    )

logger = None
   
def evaluate(
        checkpoint_folder_path,
        device,
        batch_size,
        num_workers,
        max_length,
        num_beams,
        logging_level,
        task_name,
        gpt4_nli_examples_filepaths,
        plot_confusion_matrix=False,
        sns_font_scale=1.8,
        font_size=25,
        return_outputs=False,
    ):

    global logger
    if logger is None:
        logger = get_console_logger(logging_level)

    if task_name == Seq2SeqTaskNames.NLI:
        radnli_dev_input_texts, radnli_dev_output_texts = load_radnli_dev_data(nli1_only=True)
        radnli_test_input_texts, radnli_test_output_texts = load_radnli_test_data(nli1_only=True)
        mscxrt_input_texts, mscxrt_output_texts = load_ms_cxr_t_temporal_sentence_similarity_v1_data(nli1_only=True)
        sentences = radnli_dev_input_texts + radnli_test_input_texts + mscxrt_input_texts
        if gpt4_nli_examples_filepaths is not None:
            gpt4_nli_input_texts, gpt4_nli_output_texts = load_gpt4_nli_examples_filepaths(gpt4_nli_examples_filepaths, nli1_only=True)
            sentences += gpt4_nli_input_texts
    else:
        raise NotImplementedError()

    gen_outputs, unprocessed_sentences = apply_seq2seq_model_to_sentences(
        checkpoint_folder_path=checkpoint_folder_path,
        sentences=sentences,
        logger=logger,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_length=max_length,
        num_beams=num_beams,
        postprocess_input_output_func=lambda _, output: output,
        save_outputs=False,
    )
    assert len(unprocessed_sentences) == 0

    if task_name == Seq2SeqTaskNames.NLI:
        offset = 0
        radnli_dev_gen_outputs = gen_outputs[offset:offset+len(radnli_dev_input_texts)]
        offset += len(radnli_dev_input_texts)
        radnli_test_gen_outputs = gen_outputs[offset:offset+len(radnli_test_input_texts)]
        offset += len(radnli_test_input_texts)
        mscxrt_gen_outputs = gen_outputs[offset:offset+len(mscxrt_input_texts)]
        offset += len(mscxrt_input_texts)
        if gpt4_nli_examples_filepaths is not None:
            gpt4_nli_gen_outputs = gen_outputs[offset:offset+len(gpt4_nli_input_texts)]
            offset += len(gpt4_nli_input_texts)
        assert offset == len(gen_outputs)
        assert len(radnli_dev_gen_outputs) == len(radnli_dev_output_texts)
        assert len(radnli_test_gen_outputs) == len(radnli_test_output_texts)
        assert len(mscxrt_gen_outputs) == len(mscxrt_output_texts)
        if gpt4_nli_examples_filepaths is not None:
            assert len(gpt4_nli_gen_outputs) == len(gpt4_nli_output_texts)
        
        output2label = {
            'Most likely: entailment': 0,
            'Most likely: neutral': 1,
            'Most likely: contradiction': 2,
        }

        if plot_confusion_matrix:
            from sklearn.metrics import confusion_matrix
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set(font_scale=sns_font_scale)
            label_names = ['entailment', 'neutral', 'contradiction']

        aux = [
            (radnli_dev_gen_outputs, radnli_dev_output_texts, 'RadNLI dev set'),
            (radnli_test_gen_outputs, radnli_test_output_texts, 'RadNLI test set'),
            (mscxrt_gen_outputs, mscxrt_output_texts, 'MS_CXR_T test set'),
        ]
        if gpt4_nli_examples_filepaths is not None:
            aux.append(
                (gpt4_nli_gen_outputs, gpt4_nli_output_texts, 'GPT-4 NLI examples'))
        for gen_outputs, output_texts, dataset_name in aux:
            print_blue(f'--- {dataset_name} ---', bold=True)
            pred_labels = np.array([output2label[output] for output in gen_outputs])
            gt_labels = np.array([output2label[output] for output in output_texts])
            accuracy = (pred_labels == gt_labels).mean()
            print_bold(f'Accuracy: {accuracy}')
            if plot_confusion_matrix:
                print_bold('Confusion matrix:')
                # Plot a confusion matrix
                cm = confusion_matrix(gt_labels, pred_labels)
                # Use label names instead of indices
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_names, yticklabels=label_names)
                plt.xlabel('Predicted', fontsize=font_size)
                plt.ylabel('True', fontsize=font_size)
                plt.show()

        if return_outputs:
            return {
                'radnli_dev_input_texts': radnli_dev_input_texts,
                'radnli_dev_gen_outputs': radnli_dev_gen_outputs,
                'radnli_dev_output_texts': radnli_dev_output_texts,
                'radnli_test_input_texts': radnli_test_input_texts,
                'radnli_test_gen_outputs': radnli_test_gen_outputs,
                'radnli_test_output_texts': radnli_test_output_texts,
                'mscxrt_input_texts': mscxrt_input_texts,
                'mscxrt_gen_outputs': mscxrt_gen_outputs,
                'mscxrt_output_texts': mscxrt_output_texts,
            }
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    args = parse_args()
    args = parsed_args_to_dict(args)
    evaluate(**args)