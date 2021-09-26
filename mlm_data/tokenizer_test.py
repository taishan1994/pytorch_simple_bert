from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('../../../model_hub/hfl_chinese-bert-wwm-ext/vocab.txt')

text_a = "巴黎是[MASK]国的首都。"
inputs = tokenizer.encode_plus(
    text = text_a,
    return_token_type_ids=True,
    return_attention_mask=True,
)
print(tokenizer.convert_ids_to_tokens(inputs['input_ids']))