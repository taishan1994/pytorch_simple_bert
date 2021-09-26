import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging

logging.basicConfig(level=logging.INFO)
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('../mlm/bert-base-chinese/')
# Tokenize input
text = "你叫什么名字"
tokenized_text = tokenizer.tokenize(text)
# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 2
tokenized_text[masked_index] = '[MASK]'
# Convert token to vocabulary indices,这个词汇转数字的表是预先定义好了的
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
segments_ids = [0, 0, 0, 1, 1, 1]  # [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

model = BertModel.from_pretrained('../mlm/bert-base-chinese/')
# Set the model in evaluation mode to deactivate the DropOut modules
# This is IMPORTANT to have reproducible results during evaluation!
model.eval()
# If you have a GPU, put everything on cuda
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')
# Predict hidden states features for each layer
with torch.no_grad():
    # See the models docstrings for the detail of the inputs
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)  # torch.Size([1, 14, 768]),torch.Size([1, 768])
    # Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer of the Bert model
    # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)
    encoded_layers = outputs[0]
# 加载预训练模型（权重）
model = BertForMaskedLM.from_pretrained('../mlm/bert-base-chinese/')
model.eval()
# 如果你有GPU，把所有东西都放在cuda上
tokens_tensor = tokens_tensor.to('cuda')
segments_tensors = segments_tensors.to('cuda')
model.to('cuda')
# 预测所有标记
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]
# 确认我们能预测“henson”
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)