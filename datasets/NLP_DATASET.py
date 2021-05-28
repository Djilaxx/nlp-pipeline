import torch
from torch.utils.data import Dataset

class NLP_DATASET(Dataset):
    def __init__(self, model_name, task, text, max_len, labels=None, tokenizer=None, feature_eng=None):
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
        # USING FEATURE_ENG FUNCTION TO PRE PROCESS TEXT
        if self.feature_eng is not None:
            text = self.feature_eng(text)
        # USING TOKENIZERS ENCODING TO GET TEXT DATA IN CORRECT FORMAT
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

        # GETTING ALL DATA NEEDED FOR TRANSFORMERS TRAINING
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        if self.labels is not None:
            # LABELS DATA TYPE DEPENDING ON TASK
            if self.task == "CLASSIFICATION":
                labels = torch.tensor(self.labels[index], dtype=torch.long)
            elif self.task == "REGRESSION":
                labels = torch.tensor(self.labels[index], dtype=torch.float32)

            # DISTILBERT & ROBERTA DON'T NEED TOKEN_TYPE_IDS
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'masks': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'labels': labels
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'masks': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            }
