import torch
import torch.nn as nn
import transformers

class DISTILBERT(torch.nn.Module):
    def __init__(self, task, model_config_path="distilbert-base-uncased", n_class=2):
        super(DISTILBERT, self).__init__()
        self.distilbert = transformers.DistilBertModel.from_pretrained(model_config_path)
        self.drop = nn.Dropout(0.3)
        if task == "REGRESSION":
            self.l0 = nn.Linear(768, 1)
        elif task == "CLASSIFICATION":
            self.l0 = nn.Linear(768, n_class)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask):
        output  = self.distilbert(ids, mask)
        hidden_state = output[0]
        pooled = hidden_state[:, 0]
        out = self.drop(pooled)
        out = self.l0(out)
        return out