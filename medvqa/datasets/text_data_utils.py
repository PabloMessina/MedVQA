from torch.utils.data import Dataset, DataLoader

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
        batch_dict['input_ids'] = encoding.input_ids
        batch_dict['attention_mask'] = encoding.attention_mask
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