import torch
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, data_file, seq_len, tokenizer):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        # Use tokenized data from tokenizer if available
        # try:
        #     self.data = self.tokenizer.tokens_to_indices(self.tokenizer.tokenized_data)
        # except:
        #     # Generate tokens indices of the data
        #     with open(data_file, encoding='utf-8') as f:
        #         data = f.read()
        #     print("Tokenizing data file...")
        #     self.data = self.tokenizer.encode(data, tqdm=True)      

        # Directly encode the entire data file into token indices
        with open(data_file, encoding='utf-8') as f:
            raw_text = f.read()

        print("Tokenizing data file...")
        self.data = self.tokenizer.encode(raw_text)
                               
        
        self.total_tokens = len(self.data)                                          
        self.epoch_len = self.total_tokens // self.seq_len                          # Epoch length = total tokens // training sequence length
        print(f"{self.total_tokens} tokens created from the file. Each epoch will have {self.epoch_len} batches.")

        self.end_sample_idx = self.total_tokens-self.seq_len-1                      # last possible index that can be sampled (-1 to ensure target is also present)

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, idx):
        start_idx = random.randint(0, self.end_sample_idx)                          # Select a random starting point between the start and the last possible token (due to the token length)
        idx = self.data[start_idx:start_idx+self.seq_len+1]                         # Sample data with 1 extra token which will become target y for last input token
        x, y = idx[:-1], idx[1:]                                                    # Shift x and y to form input and target
        return torch.LongTensor(x), torch.LongTensor(y)


def collate_fn(data):
    """
    function to format the data to form a batch with x and y

    Input:
        data: torch tensors with x and y as list of tuples
    
    Returns:
        Tensors: x and y
    """    

    x, y = zip(*data)
    x = torch.stack(x)
    y = torch.stack(y)
    return x, y


def get_dataloader(data_file, batch_size, seq_len, n_workers, tokenizer):
    dataset    = CustomDataset(data_file, seq_len, tokenizer)                                                               # Create Dataset from the file using tokenizer
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, collate_fn=collate_fn)     # Create Dataloader for the dataset
    return dataloader

