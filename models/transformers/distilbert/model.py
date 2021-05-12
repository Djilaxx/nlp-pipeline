import torch
import torch.nn as nn
import transformers

class distilbert(torch.nn.Module):
    def __init__(self, n_class, model_config_path):
        super(distilbert, self).__init__()
        self.distill_bert = transformers.DistilBertModel.from_pretrained(model_config_path)
        self.drop = nn.Dropout(0.3)
        self.l0 = nn.Linear(768, n_class)
        torch.nn.init.normal_(self.l0.weight, std=0.02)
    
    def forward(self, ids, mask):
        dsb_output  = self.distill_bert(ids, mask)
        hidden_state = dsb_output[0]
        pooled = hidden_state[:, 0]
        out = self.drop(pooled)
        out = self.l0(out)
        return out