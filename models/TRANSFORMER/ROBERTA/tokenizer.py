from transformers import RobertaTokenizer

def tokenizer():
    return RobertaTokenizer.from_pretrained("roberta-base")