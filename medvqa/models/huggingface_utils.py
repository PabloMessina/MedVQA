from tqdm import tqdm
import numpy as np
from medvqa.datasets.text_data_utils import create_text_dataset_and_dataloader
from medvqa.models.checkpoint import get_checkpoint_filepath

def _adapt_checkpoint_keys(checkpoint):
    for key in list(checkpoint.keys()):
        if key.startswith('model.'):
            checkpoint[key[6:]] = checkpoint.pop(key)
    return checkpoint

def compute_text_embeddings(model_url, get_tokenizer_func, texts, device, batch_size=32, num_workers=0,
                            model_checkpoint_folder_path=None):
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'CPU')

    # Load model
    model = AutoModel.from_pretrained(model_url, trust_remote_code=True)
    model.to(device)

    # Load pre-trained weights from checkpoint folder (if provided)
    if model_checkpoint_folder_path is not None:
        model_checkpoint_filepath = get_checkpoint_filepath(model_checkpoint_folder_path)
        print(f'Loading model weights from {model_checkpoint_filepath}')
        checkpoint = torch.load(model_checkpoint_filepath, map_location=device)
        model.load_state_dict(_adapt_checkpoint_keys(checkpoint['model']), strict=False)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_url, trust_remote_code=True)
    tokenizer_func = get_tokenizer_func(tokenizer)

    # Create dataset and dataloader
    _, dataloader = create_text_dataset_and_dataloader(
        texts=texts,
        batch_size=batch_size,
        num_workers=num_workers,
        tokenizer_func=tokenizer_func,
    )

    # Run inference
    model.eval()
    embeddings = np.zeros((len(texts), model.config.projection_size), dtype=np.float32)
    offset = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader), mininterval=2):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_embeddings = model.get_projected_text_embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask)
            batch_size = batch_embeddings.shape[0]
            embeddings[offset:offset+batch_size] = batch_embeddings.cpu().numpy()
            offset += batch_size
    assert offset == len(texts)

    # Cleanup
    del model
    del tokenizer
    del dataloader
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return embeddings

def _get_microsoft_BERT_tokenizer_func(tokenizer):
    return lambda x: tokenizer.batch_encode_plus(batch_text_or_text_pairs=x,
                                                add_special_tokens=True,
                                                padding='longest',
                                                return_tensors='pt')

def compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized(texts, device, batch_size=32, num_workers=0,
                                                                model_checkpoint_folder_path=None):
    return compute_text_embeddings(
        model_url='microsoft/BiomedVLP-CXR-BERT-specialized',
        get_tokenizer_func=_get_microsoft_BERT_tokenizer_func,
        texts=texts,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        model_checkpoint_folder_path=model_checkpoint_folder_path,
    )

def compute_text_embeddings_with_BiomedVLP_BioVilT(texts, device, batch_size=32, num_workers=0,
                                                                model_checkpoint_folder_path=None):
    return compute_text_embeddings(
        model_url='microsoft/BiomedVLP-BioViL-T',
        get_tokenizer_func=_get_microsoft_BERT_tokenizer_func,
        texts=texts,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        model_checkpoint_folder_path=model_checkpoint_folder_path,
    )