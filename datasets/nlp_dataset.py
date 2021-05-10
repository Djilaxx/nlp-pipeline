import torch
from torch.utils.data import Dataset, DataLoader
from utils.clean_text import pre_process_text

class NLP_DATASET(Dataset):
    def __init__(self, model_name, text, labels, max_len, tokenizer = None, feature_eng = None):
        self.model_name = model_name
        self.text = text
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.feature_eng = feature_eng

    #RETURN THE LENGHT OF THE DATASET
    def __len__(self):
        return len(self.text)
    
    #FUNCTION THAT RETURN ONE DATAPOINT (INPUT + LABEL)
    def __getitem__(self, index):
        # LIST WHERE ONE ROW OF TEXT DATA
        text = str(self.text[index])

        if feature_eng is not None:
            text = feature_eng(text)

        if tokenizer is not None:
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
        
class NLP_QA_DATASET(Dataset):
    def __init__(self, model_name, text, labels, max_len, tokenizer = None, feature_eng = None):
        self.model_name = model_name
        self.text = text
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.feature_eng = feature_eng
    
    #RETURN THE LENGHT OF THE DATASET
    def __len__(self):
        return len(self.text)
    
    #FUNCTION THAT RETURN ONE DATAPOINT (INPUT + LABEL)
    def __getitem__(self, index):
        
        # LIST WHERE ONE ROW OF TEXT DATA
        text = str(self.text[index])

        if feature_eng is not None:
            text = feature_eng(text)

        
        if tokenizer is not None:
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