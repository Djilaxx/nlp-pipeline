import torch
from torch.utils.data import Dataset, DataLoader
from utils.clean_text import pre_process_text

class TEXT_DS(Dataset):
    def __init__(self, model_name, text, labels, max_len, tokenizer = None):
        self.model_name = model_name
        self.text = text
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        text = str(self.text[index])
        text = pre_process_text(text)
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'masks': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(self.labels[index], dtype=torch.long)
        }
        
    def __len__(self):
        return len(self.text)