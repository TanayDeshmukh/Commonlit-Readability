import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from utils import tokenizer, PAD, EOS
from torch.nn.utils.rnn import pad_sequence

class CommonlitDataset(Dataset):
    def __init__(self, root_dir, split):

        self.root_dir = root_dir
        self.split = split
        
        self.text_ids = []
        self.text = []
        self.targets = []
        self.std_err = []
        
        if split == 'train':
            self.load_data_train()
            self.load_fn = self.get_item_train
        else:
            self.load_data_test()
            self.load_fn = self.get_item_test
    
    def load_data_train(self):
        
        data_file = os.path.join(self.root_dir, self.split+'.csv')
        df = pd.read_csv(data_file)
        for i, row in df.iterrows():
            self.text_ids.append(row['id'])
            self.text.append(tokenizer.encode(row['excerpt']))
            self.targets.append(row['target'])
            self.std_err.append(row['standard_error'])
    
    def load_data_test(self):        
        data_file = os.path.join(self.root_dir, self.split+'.csv')
        df = pd.read_csv(data_file)
        for i, row in df.iterrows():
            self.text_ids.append(row['id'])
            self.text.append(tokenizer.encode(row['excerpt']))
   
    def get_item_train(self, idx):
        text_id = self.text_ids[idx]
        text = self.text[idx]
        text = text + [EOS]
        text = torch.tensor(text, dtype=torch.long)

        target = self.targets[idx]
        std_err = self.std_err[idx]

        return text_id, text, target, std_err

    def get_item_test(self, idx):
        text_id = self.text_ids[idx]
        text = self.text[idx]
        text = text + [EOS]
        text = torch.tensor(text, dtype=torch.long)

        return text_id, text
    
    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        return self.load_fn(idx)

def collate_fn_train(batch):

    batch = list(zip(*batch))

    text_id = batch[0]
    text = pad_sequence(batch[1], batch_first=True, padding_value=PAD)
    target = torch.tensor(batch[2], dtype=torch.float)
    std_err = torch.tensor(batch[3], dtype=torch.float)

    return text_id, text, target, std_err


def collate_fn_test(batch):

    batch = list(zip(*batch))

    text_id = batch[0]
    text = pad_sequence(batch[1], batch_first=True, padding_value=PAD)

    return text_id, text