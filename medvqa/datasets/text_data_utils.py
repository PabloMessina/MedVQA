from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import sent_tokenize

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i]

def get_text_collate_batch_fn(tokenizer_func):

    def collate_batch_fn(batch):
        assert type(batch) == list, type(batch)
        assert len(batch) > 0, len(batch)
        assert type(batch[0]) == str, type(batch[0])
        # encoding = tokenizer(
        #     batch,
        #     padding="longest",
        #     return_tensors="pt",
        # )
        encoding = tokenizer_func(batch)
        batch_dict = {}
        batch_dict['encoding'] = encoding
        return batch_dict
    
    return collate_batch_fn

def create_text_dataset_and_dataloader(texts, batch_size, num_workers, tokenizer_func):
    # Create collate batch function
    collate_batch_fn = get_text_collate_batch_fn(tokenizer_func)
    # Create dataset
    dataset = TextDataset(texts)
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_batch_fn,
        pin_memory=True,
    )
    return dataset, dataloader

def split_text_into_chunks(text, max_length):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ''
    for sentence in sentences:
        if len(chunk) + len(sentence) > max_length and chunk != '':
            chunks.append(chunk)
            chunk = ''
        if chunk != '':
            if chunk[-1] != '.':
                chunk += '. '
            else:
                chunk += ' '
        chunk += sentence
    if chunk != '':
        chunks.append(chunk)
    return chunks

def sentence_tokenize_texts_in_parallel(texts, num_workers=None):
    import multiprocessing as mp
    if num_workers is None:
        num_workers = mp.cpu_count()
    else:
        num_workers = min(num_workers, mp.cpu_count())
    with mp.Pool(num_workers) as pool:
        sentences = pool.map(sent_tokenize, texts)
    return sentences

