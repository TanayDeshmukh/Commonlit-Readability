from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
PAD = tokenizer.pad_token_id
EOS = tokenizer.convert_tokens_to_ids('.')