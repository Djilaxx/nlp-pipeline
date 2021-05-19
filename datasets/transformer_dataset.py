import torch
from torch.utils.data import Dataset, DataLoader

class NLP_DATASET(Dataset):
    def __init__(self, model_name, task, text, labels, max_len, tokenizer = None, feature_eng = None):
        self.model_name = model_name
        self.task = task
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

        if self.feature_eng is not None:
            text = self.feature_eng(text)

        if self.tokenizer is not None:
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
        token_type_ids = inputs["token_type_ids"]
        if self.task == "CL":
            labels = torch.tensor(self.labels[index], dtype=torch.long)
        elif self.task == "REG":
            labels = torch.tensor(self.labels[index], dtype=torch.float32)

        if self.model_name in ["DISTILBERT", "ROBERTA"]:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'masks': torch.tensor(mask, dtype=torch.long),
                'labels': labels
            }
        elif self.model_name in ["BERT"]:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'masks': torch.tensor(mask, dtype=torch.long),
                'token_type_ids' : torch.tensor(token_type_ids, dtype=torch.long),
                'labels': labels
            }
