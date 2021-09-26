import torch
from transformers import BertModel, BertTokenizer, BertForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pretrained_path = '../../model_hub/hfl_chinese-bert-wwm-ext/'
tokenizer = BertTokenizer.from_pretrained(pretrained_path)
# robertaModel = BertModel.from_pretrained(pretrained_path)
# for name, param in robertaModel.named_parameters():
#     print(name)
# robertaModel.eval()
# robertaModel.to(device)
#
# text = '巴黎是[MASK]国的首都。'
# input = tokenizer.encode_plus(text=text,
#                               max_length=512,
#                               padding='max_length',
#                               truncation='only_first',
#                               return_token_type_ids=True,
#                               return_attention_mask=True,
#                               return_tensors='pt', )
# input_token = input['input_ids'].to(device)
# segment_ids = input['token_type_ids'].to(device)
# attention_mask = input['attention_mask'].to(device)
# with torch.no_grad():
#     outputs = robertaModel(input_token, segment_ids, attention_mask)
#     # outputs = outputs[0]
#     print(outputs[0].shape)
#     print(outputs[1].shape)

robertaModel = BertForMaskedLM.from_pretrained(pretrained_path)
robertaModel = robertaModel.to(device)
robertaModel.eval()
for name in robertaModel.state_dict():
    print(name, robertaModel.state_dict()[name].shape)

text = '巴黎是[MASK]国的首都。'
# text = "你[MASK]什么名字"
# tokens = tokenizer.tokenize(text)
input = tokenizer.encode_plus(text=text,
                              return_token_type_ids=True,
                              return_attention_mask=True,
                              return_tensors='pt', )
print(input['input_ids'][0])
tokens = tokenizer.convert_ids_to_tokens(input['input_ids'][0])
print(tokens[4])
input_token = input['input_ids'].to(device)
segment_ids = input['token_type_ids'].to(device)
attention_mask = input['attention_mask'].to(device)
ind = tokens.index('[MASK]')
print(ind)
for key in robertaModel.state_dict():
    print(key)
with torch.no_grad():
    outputs = robertaModel(input_token, segment_ids, attention_mask)
    outputs = outputs[0] # torch.Size([1, 512, 21128])
    print(outputs.shape)
    outputs = outputs[0][ind, :]
    print(outputs.shape)

    outputs = torch.argmax(outputs, dim=-1)
    print(outputs)
    words = tokenizer.convert_ids_to_tokens(outputs.item())
    print(words)
    # outputs = torch.topk(outputs, 5, dim=-1)
    # outputs = outputs[1].detach().cpu().numpy()
    # words = tokenizer.convert_ids_to_tokens(outputs)
    # print(words)
