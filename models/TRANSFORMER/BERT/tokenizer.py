from transformers import BertTokenizer

def tokenizer():
    return BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
