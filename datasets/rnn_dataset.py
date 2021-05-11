import torch
from torch.utils.data import Dataset, DataLoader

class RNN_DATASET(Dataset):
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

        if self.feature_eng is not None:
            text = self.feature_eng(text)

        # EMBEDDINGS (WORD2VEC - GLOVE - ELMO)

        # RETURN EMBEDDINGS + TEXT + LABELS