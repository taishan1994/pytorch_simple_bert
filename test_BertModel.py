import torch
import torch.nn as nn
from transformers import BertTokenizer

from Transformer import Transformer
from BertEmbeddings import BertEmbeddings
from Pooler import BertPooler
from BertOnlyMLMHead import BertOnlyMLMHead

class Config:
    vocab_size = 21128
    hidden_size = 768
    attention_head_num = 12
    attention_head_size = hidden_size // attention_head_num
    assert "self.hidden必须要整除self.attention_heads"
    intermediate_size = 3072
    num_hidden_layers = 12
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    AttentionMask = True
    max_len = 512
    layer_norm_eps = 1e-12
    hidden_act = "gelu"

class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.attention_head_num = config.attention_head_num
        self.attention_head_size = config.attention_head_size
        self.intermediate_size = config.intermediate_size
        self.num_hidden_layers = config.num_hidden_layers
        self.device = config.device
        self.AttentionMask = config.AttentionMask
        self.max_len = config.max_len
        # 申明网络
        self.roberta_emd = BertEmbeddings(vocab_size=self.vocab_size, max_len=self.max_len,
                                             hidden_size=self.hidden_size, device=self.device)
        self.transformer_blocks = nn.ModuleList(
            Transformer(
                hidden_size=self.hidden_size,
                attention_head_num=self.attention_head_num,
                attention_head_size=self.attention_head_size,
                intermediate_size=self.intermediate_size).to(self.device)
            for _ in range(self.num_hidden_layers)
        )
        self.pooler = BertPooler(self.hidden_size)

    def load_local2target(self):
        local2target_emb = {
            'roberta_emd.token_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
            'roberta_emd.type_embeddings.weight': 'bert.embeddings.token_type_embeddings.weight',
            'roberta_emd.position_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
            'roberta_emd.emb_normalization.weight': 'bert.embeddings.LayerNorm.weight',
            'roberta_emd.emb_normalization.bias': 'bert.embeddings.LayerNorm.bias'
        }

        local2target_transformer = {
            'transformer_blocks.%s.multi_attention.q_dense.weight': 'bert.encoder.layer.%s.attention.self.query.weight',
            'transformer_blocks.%s.multi_attention.q_dense.bias': 'bert.encoder.layer.%s.attention.self.query.bias',
            'transformer_blocks.%s.multi_attention.k_dense.weight': 'bert.encoder.layer.%s.attention.self.key.weight',
            'transformer_blocks.%s.multi_attention.k_dense.bias': 'bert.encoder.layer.%s.attention.self.key.bias',
            'transformer_blocks.%s.multi_attention.v_dense.weight': 'bert.encoder.layer.%s.attention.self.value.weight',
            'transformer_blocks.%s.multi_attention.v_dense.bias': 'bert.encoder.layer.%s.attention.self.value.bias',
            'transformer_blocks.%s.multi_attention.o_dense.weight': 'bert.encoder.layer.%s.attention.output.dense.weight',
            'transformer_blocks.%s.multi_attention.o_dense.bias': 'bert.encoder.layer.%s.attention.output.dense.bias',
            'transformer_blocks.%s.attention_layernorm.weight': 'bert.encoder.layer.%s.attention.output.LayerNorm.weight',
            'transformer_blocks.%s.attention_layernorm.bias': 'bert.encoder.layer.%s.attention.output.LayerNorm.bias',
            'transformer_blocks.%s.feedforward.dense1.weight': 'bert.encoder.layer.%s.intermediate.dense.weight',
            'transformer_blocks.%s.feedforward.dense1.bias': 'bert.encoder.layer.%s.intermediate.dense.bias',
            'transformer_blocks.%s.feedforward.dense2.weight': 'bert.encoder.layer.%s.output.dense.weight',
            'transformer_blocks.%s.feedforward.dense2.bias': 'bert.encoder.layer.%s.output.dense.bias',
            'transformer_blocks.%s.feedforward_layernorm.weight': 'bert.encoder.layer.%s.output.LayerNorm.weight',
            'transformer_blocks.%s.feedforward_layernorm.bias': 'bert.encoder.layer.%s.output.LayerNorm.bias',
        }
        local2target_pooler = {
            "pooler.dense.weight": "bert.pooler.dense.weight",
            "pooler.dense.bias": "bert.pooler.dense.bias",
        }

        return local2target_emb, local2target_transformer, local2target_pooler

    def load_pretrain(self, sen_length, path):
        local2target_emb, local2target_transformer, local2target_pooler = self.load_local2target()
        pretrain_model_dict = torch.load(path)
        if sen_length == 512:
            finetune_model_dict = self.state_dict()
            new_parameter_dict = {}
            # 加载embedding层参数
            for key in local2target_emb:
                local = key
                target = local2target_emb[key]
                new_parameter_dict[local] = pretrain_model_dict[target]
            # 加载transformerblock层参数
            for i in range(self.num_hidden_layers):
                for key in local2target_transformer:
                    local = key % i
                    target = local2target_transformer[key] % i
                    new_parameter_dict[local] = pretrain_model_dict[target]
            for key in local2target_pooler:
                local = key
                target = local2target_pooler[key]
                new_parameter_dict[local] = pretrain_model_dict[target]
            finetune_model_dict.update(new_parameter_dict)
            self.load_state_dict(finetune_model_dict)
        else:
            raise Exception('请输入预训练模型正确的长度')

    def gen_attention_masks(self, attention_mask):
        """

        :param segment_ids:
        :return:[batchsize, max_len, max_len]
        """
        size = list(attention_mask.size())
        batch = size[0]
        max_len = size[1]
        process_attention_mask = torch.zeros(batch, max_len, max_len, requires_grad=False)
        true_len = torch.sum(attention_mask, dim=1)
        for i in range(batch):
            process_attention_mask[i, :true_len[i], :true_len[i]] = 1
        return process_attention_mask

    def forward(self, input_token, segment_ids, attention_mask):
        embedding_x = self.roberta_emd(input_token, segment_ids)
        if self.AttentionMask:
            attention_mask = self.gen_attention_masks(attention_mask).to(self.device)
        else:
            attention_mask = None
        feedforward_x = None
        # transformer
        for i in range(self.num_hidden_layers):
            if i == 0:
                feedforward_x = self.transformer_blocks[i](embedding_x, attention_mask)
            else:
                feedforward_x = self.transformer_blocks[i](feedforward_x, attention_mask)
        sequence_output = feedforward_x
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


if __name__ == '__main__':
    config = Config()
    model = TransformerModel(config)
    model.load_pretrain(512, '../../model_hub/hfl_chinese-bert-wwm-ext/pytorch_model.bin')
    model.eval()
    model.to(config.device)
    for name, param in model.named_parameters():
        print(name)
    tokenizer = BertTokenizer.from_pretrained('../../model_hub/hfl_chinese-bert-wwm-ext/')
    # text = '在数据脱敏比赛或者某些垂类领域中，使用该领域的文本继续预训练，往往可以取得一个更好的结果。这篇文章主要讲我目前使用过的两种预训练方法。'
    text = '在数据脱敏比赛或者某些垂类领域中，使用该领域的文本继续预训练，往往可以取得一个更好的结果。这篇文章主要讲我目前使用过的两种预训练方法。'
    input = tokenizer.encode_plus(text=text,
                                  max_length=256,
                                  padding='max_length',
                                  truncation='only_first',
                                  return_token_type_ids=True,
                                  return_attention_mask=True,
                                  return_tensors='pt', )
    print(tokenizer.convert_ids_to_tokens(input['input_ids'][0]))
    input_token = input['input_ids'].to(config.device)
    segment_ids = input['token_type_ids'].to(config.device)
    attention_mask = input['attention_mask'].to(config.device)
    print(input_token)
    with torch.no_grad():
        sequence_output, pooled_output = model(input_token, segment_ids, attention_mask)
        print(sequence_output.shape)
        print(pooled_output.shape)
        # outputs = torch.topk()
