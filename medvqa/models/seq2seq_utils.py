import os
import torch
import numpy as np
from medvqa.datasets.text_data_utils import create_text_dataset_and_dataloader
from medvqa.models.checkpoint import get_checkpoint_filepath, load_metadata
from medvqa.models.nlp.seq2seq import Seq2SeqModel
from transformers import AutoTokenizer
from tqdm import tqdm
from medvqa.utils.common import get_timestamp

from medvqa.utils.files_utils import save_jsonl

def apply_seq2seq_model_to_sentences(
        checkpoint_folder_path, sentences, device, batch_size, num_workers, max_length, num_beams,
        postprocess_input_output_func, save_outputs=True, save_dir=None, save_filename_prefix=None,
        return_checkpoint_path=False, logger=None):

    if logger is None:
        logger_info = print
        logger_warning = print
    else:
        logger_info = logger.info
        logger_warning = logger.warning

    # Print example sentences
    logger_info(f"Example sentences to process:")
    indices = np.random.choice(len(sentences), min(10, len(sentences)), replace=False)
    for i in indices:
        logger_info(f"{i}: {sentences[i]}")
    
    # Load model metadata
    metadata = load_metadata(checkpoint_folder_path)
    model_kwargs = metadata['model_kwargs']
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and device in ['cuda', 'gpu', 'GPU'] else 'cpu')
    
    # Create model
    logger_info(f"Creating Seq2SeqModel")
    model = Seq2SeqModel(**model_kwargs)
    model = model.to(device)

    # Load model weights
    logger_info(f"Loading model weights from {checkpoint_folder_path}")
    checkpoint_path = get_checkpoint_filepath(checkpoint_folder_path)
    logger_info(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])

    # Create tokenizer
    logger_info(f"Creating tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_kwargs['model_name'])

    # Create dataloader
    logger_info(f"Creating dataloader")
    tokenizer_func = lambda x: tokenizer(x, padding="longest", return_tensors="pt")
    _, dataloader = create_text_dataset_and_dataloader(
        texts=sentences,
        batch_size=batch_size,  
        num_workers=num_workers,
        tokenizer_func=tokenizer_func,
    )

    # Run inference
    logger_info(f"Running inference")
    model.eval()
    outputs = [None] * len(sentences)
    idx = 0
    unprocessed_sentences = []
    print_counts = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), mininterval=2):
            encoding = batch['encoding']
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            output_ids = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_len=max_length,
                num_beams=num_beams,
                mode='test',
            )
            output_texts_batch = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for output_text in output_texts_batch:
                try:
                    outputs[idx] = postprocess_input_output_func(sentences[idx], output_text)
                    if (idx-1) % 5000 == 0 and print_counts < 4:
                        logger_info(f"Processed {idx} sentences")
                        logger_info(f"Example output:")
                        logger_info(f"{outputs[idx-1]}")
                        print_counts += 1
                except Exception as e:
                    logger_warning(f"Failed to process output text: {output_text}, for sentence: {sentences[idx]}, idx: {idx}")
                    logger_warning(f"Exception: {e}")
                    unprocessed_sentences.append(sentences[idx])
                idx += 1
    assert idx == len(sentences)
    
    outputs = [x for x in outputs if x is not None]
    
    if len(unprocessed_sentences) > 0:
        logger_warning(f"Failed to process {len(unprocessed_sentences)} sentences")
    
    logger_info(f"Successfully processed {len(outputs)} sentences")
    logger_info(f"Example output:")
    logger_info(f"{outputs[-1]}")

    assert len(sentences) == len(outputs) + len(unprocessed_sentences)
    
    if save_outputs:
        assert save_dir is not None
        assert save_filename_prefix is not None

        # Save outputs
        timestamp = get_timestamp()
        save_filename = f"{save_filename_prefix}_{model.get_name()}_{max_length}_{num_beams}_{timestamp}.jsonl"
        save_filepath = os.path.join(save_dir, save_filename)
        logger_info(f"Saving outputs to {save_filepath}")
        save_jsonl(outputs, save_filepath)

        # Save unprocessed sentences
        if len(unprocessed_sentences) > 0:
            save_filepath = save_filepath[:-7] + ".unprocessed.jsonl" # replace .jsonl with .unprocessed.jsonl
            logger_info(f"Saving unprocessed sentences to {save_filepath}")
            save_jsonl(unprocessed_sentences, save_filepath)

    logger_info(f"DONE")

    if return_checkpoint_path:
        return outputs, unprocessed_sentences, checkpoint_path

    return outputs, unprocessed_sentences