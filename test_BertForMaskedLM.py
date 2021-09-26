import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm
import numpy as np

from Transformer import Transformer
from BertEmbeddings import BertEmbeddings
from BertOnlyMLMHead import BertOnlyMLMHead


class MyBertForMaskedLM(nn.Module):
    def __init__(self, config):
        super(MyBertForMaskedLM, self).__init__()
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
        self.cls = BertOnlyMLMHead(config)

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
        local2target_cls = {
            "cls.predictions.bias": "cls.predictions.bias",
            "cls.predictions.transform.dense.weight": "cls.predictions.transform.dense.weight",
            "cls.predictions.transform.dense.bias": "cls.predictions.transform.dense.bias",
            "cls.predictions.transform.LayerNorm.weight": "cls.predictions.transform.LayerNorm.weight",
            "cls.predictions.transform.LayerNorm.bias": "cls.predictions.transform.LayerNorm.bias",
            "cls.predictions.decoder.weight": "cls.predictions.decoder.weight",
        }

        return local2target_emb, local2target_transformer, local2target_cls

    def load_pretrain(self, sen_length, path):
        local2target_emb, local2target_transformer, local2target_cls = self.load_local2target()
        pretrain_model_dict = BertForMaskedLM.from_pretrained(path).state_dict()
        if sen_length == 512:
            finetune_model_dict = self.state_dict()
            # print(pretrain_model_dict.keys())
            # print(finetune_model_dict.keys())
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
            for key in local2target_cls:
                local = key
                target = local2target_cls[key]
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

    def forward(self,
                input_token,
                segment_ids,
                attention_mask,
                ):
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
        # print("feedforward_x.shape:",feedforward_x.shape)

        sequence_output = self.cls(feedforward_x)
        # print("sequence_output.shape", sequence_output.shape)
        return sequence_output


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
    max_len = 512  # 加载预训练模型的长度
    layer_norm_eps = 1e-12
    hidden_act = "gelu"

    train_file = './mlm_data/mlm_data.txt'
    train_max_len = 64  # 实际训练的最大长度
    train_epoch = 10
    bert_dir = '../../model_hub/hfl_chinese-bert-wwm-ext/'
    tokenizer = BertTokenizer.from_pretrained(bert_dir + 'vocab.txt')
    output_dir = './checkpoints/'
    use_pretrained = True
    batch_size = 128
    max_predictions_per_seq = 6
    lr = 2e-5


class MLMDataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.examples = self.get_data()
        self.nums = len(self.examples)

    def get_data(self):
        examples = []
        with open(self.config.train_file, 'r') as fp:
            lines = fp.read().strip().split('\n')
            for i, line in tqdm(enumerate(lines)):
                line = eval(line)
                tokens = line['tokens']
                # segment_ids = line['segment_ids']
                masked_lm_positions = line['masked_lm_positions']
                masked_lm_labels = line['masked_lm_labels']
                masked_lm_ids = [self.config.tokenizer.convert_tokens_to_ids(masked_lm_label)
                                 for masked_lm_label in masked_lm_labels]
                label_weight = [1] * len(masked_lm_positions)
                while len(label_weight) < self.config.max_predictions_per_seq:
                    masked_lm_positions.append(0)
                    masked_lm_ids.append(0)
                    label_weight.append(0.0)

                input_ids = self.config.tokenizer.convert_tokens_to_ids(tokens) + [0] * (
                        self.config.train_max_len - len(tokens))
                tokens_type_ids = [0] * self.config.train_max_len
                attention_mask = [1] * len(tokens) + [0] * (self.config.train_max_len - len(tokens))

                if i == 0:
                    print(input_ids)
                    print(tokens_type_ids)
                    print(attention_mask)
                    print(masked_lm_positions)
                    print(masked_lm_ids)
                    print(label_weight)
                examples.append(
                    (
                        input_ids,
                        tokens_type_ids,
                        attention_mask,
                        masked_lm_positions,
                        masked_lm_ids,
                        label_weight,
                    )
                )
        return examples

    def __len__(self):
        return self.nums

    def __getitem__(self, item):
        data = {
            "input_ids": self.examples[item][0],
            "token_type_ids": self.examples[item][1],
            "attention_mask": self.examples[item][2],
            "masked_lm_positions": self.examples[item][3],
            "masked_lm_ids":self.examples[item][4],
            "label_weight": self.examples[item][5],
        }
        for key in data:
            if key != 'label_weight':
                data[key] = torch.tensor(data[key]).long()
            else:
                data[key] = torch.tensor(data[key]).float()
        return data


class Trainer:
    def __init__(self, config, train_loader):
        self.config = config
        self.model = MyBertForMaskedLM(config)
        if config.use_pretrained:
            self.model.load_pretrain(config.max_len, config.bert_dir)
        else:
            self.model.load_state_dict(torch.load(config.output_dir+'pytorch_bin.model'))
        self.model.to(config.device)
        self.train_loader = train_loader
        self.optim = optim.Adam(self.model.parameters(), lr=self.config.lr)

    def train(self):
        self.model.train()
        total_step = self.config.train_epoch * len(self.train_loader)
        global_step = 0
        for epoch in range(self.config.train_epoch):
            for step, data in enumerate(self.train_loader):
                for key in data:
                    data[key] = data[key].to(self.config.device)
                logits = self.model(
                    data['input_ids'],
                    data['token_type_ids'],
                    data['attention_mask'],
                ) # [batchsize, train_max_len, vocab_size]
                masked_lm_positions = data['masked_lm_positions']
                masked_lm_ids = data['masked_lm_ids']
                label_weight = data['label_weight']
                batch_size = logits.shape[0]
                seq_length = logits.shape[1]
                width = logits.shape[2]
                flat_offsets = (torch.arange(0, batch_size).long() * seq_length).reshape(-1, 1).to(self.config.device)
                flat_positions = (masked_lm_positions + flat_offsets).reshape(-1).to(self.config.device)
                flat_sequence_tensor = logits.view(batch_size * seq_length, width)
                output_tensor = torch.index_select(flat_sequence_tensor, 0, flat_positions)
                log_probs = F.log_softmax(output_tensor, dim=-1)
                log_probs = log_probs.view(batch_size, -1, width)
                one_hot_ids = F.one_hot(masked_lm_ids, num_classes=self.config.vocab_size)
                per_example_loss = - torch.sum(log_probs * one_hot_ids, dim=-1)
                numerator = torch.sum(label_weight * per_example_loss, dim=-1)
                denominator = torch.sum(label_weight, dim=-1) + 1e-5
                loss = numerator / denominator
                loss = torch.sum(loss, dim=-1) / batch_size
                self.model.zero_grad()
                loss.backward()
                self.optim.step()
                print('epoch:{} step:{}/{} loss:{}'.format(epoch, global_step, total_step, loss.item()))
                global_step += 1
        torch.save(self.model.state_dict(), self.config.output_dir + 'pytorch_model.bin')

    def predict(self):
        self.model.eval()
        text = '宇[MASK]员尿液堵塞国际空间站水循环系统'
        input = self.config.tokenizer.encode_plus(
            text = text,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        for key in input:
            input[key] = input[key].to(self.config.device)
        with torch.no_grad():
            input_token = input["input_ids"]
            segment_ids = input["token_type_ids"]
            attention_mask = input["attention_mask"]
            logits = self.model(input_token, segment_ids, attention_mask)
            print(logits.shape)
            ind = 2
            logits = logits[:, ind, :]
            logits = np.argmax(logits.cpu().detach().numpy(), -1)
            print(self.config.tokenizer.convert_ids_to_tokens(logits))

if __name__ == '__main__':
    config = Config()
    # mlmDataset = MLMDataset(config)
    # train_loader = DataLoader(mlmDataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)
    # trainer = Trainer(config, train_loader)
    # trainer.train()

    trainer = Trainer(config, None)
    config.use_pretrained = False
    trainer.predict()
