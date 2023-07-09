from medvqa.datasets.text_data_utils import create_text_dataset_and_dataloader
from tqdm import tqdm
import numpy as np

def compute_text_embeddings(model_url, get_tokenizer_func, texts, device, logger, batch_size=32, num_workers=0):
    import torch
    from transformers import AutoTokenizer, AutoModel
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'CPU')

    # Load model
    model = AutoModel.from_pretrained(model_url, trust_remote_code=True)
    model.to(device)

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
    logger.info(f"Running inference")
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

    return embeddings

def compute_text_embeddings_with_BiomedVLP_CXR_BERT_specialized(texts, device, logger, batch_size=32, num_workers=0):

    def _get_tokenizer_func(tokenizer):
        return lambda x: tokenizer.batch_encode_plus(batch_text_or_text_pairs=x,
                                                     add_special_tokens=True,
                                                     padding='longest',
                                                     return_tensors='pt')

    return compute_text_embeddings(
        model_url='microsoft/BiomedVLP-CXR-BERT-specialized',
        get_tokenizer_func=_get_tokenizer_func,
        texts=texts,
        device=device,
        logger=logger,
        batch_size=batch_size,
        num_workers=num_workers,
    )