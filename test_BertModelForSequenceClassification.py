import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, classification_report

from test_BertModel import TransformerModel


class ClsDataset(Dataset):
    def __init__(self, path, tokenizer, train_max_len):
        self.path = path
        self.tokenizer = tokenizer
        self.train_max_len = train_max_len
        self.feeatures = self.get_features()
        self.nums = len(self.feeatures)

    def get_features(self):
        features = []
        with open(self.path, 'r') as fp:
            lines = fp.read().strip().split('\n')
            for i,line in enumerate(lines):
                line = line.split('\t')
                text = line[0]
                label = line[1]
                inputs = self.tokenizer.encode_plus(
                    text=text,
                    max_length=self.train_max_len,
                    padding="max_length",
                    truncation="only_first",
                    return_attention_mask=True,
                    return_token_type_ids=True,
                )
                if i < 3:
                    print("input_ids:", str(inputs['input_ids']))
                    print("token_type_ids:", str(inputs['token_type_ids']))
                    print("attention_mask:", str(inputs['attention_mask']))
                    print("label:", label)
                features.append(
                    (
                        inputs['input_ids'],
                        inputs['token_type_ids'],
                        inputs['attention_mask'],
                        int(label),
                    )
                )
        return features

    def __len__(self):
        return self.nums

    def __getitem__(self, item):
        data = {
            "token_ids": torch.tensor(self.feeatures[item][0]).long(),
            "token_type_ids": torch.tensor(self.feeatures[item][1]).long(),
            "attention_masks": torch.tensor(self.feeatures[item][2]).long(),
            "labels": torch.tensor(self.feeatures[item][3]).long(),
        }
        return data


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
    max_len = 512 # 预训练模型的句子最大长度
    layer_norm_eps = 1e-12
    hidden_act = "gelu"
    hidden_dropout_prob = 0.1

    # 以下的是训练的一些参数
    bert_dir = '../../model_hub/hfl_chinese-bert-wwm-ext/'
    train_max_len = 32
    batch_size = 64
    train_epochs = 5
    lr = 2e-5
    num_labels = 10
    output_dir = './checkpoints/'
    use_pretrained = True



class TransformerModelForSequenceClassification(nn.Module):
    def __init__(self, config):
        super(TransformerModelForSequenceClassification, self).__init__()
        self.bert = TransformerModel(config)
        if config.use_pretrained:
            self.bert.load_pretrain(config.max_len, config.bert_dir+'pytorch_model.bin')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
                input_ids,
                token_type_ids,
                attention_mask,
                ):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class Trainer:
    def __init__(self, args, train_loader, dev_loader, test_loader):
        self.args = args
        self.device = args.device
        self.model = TransformerModelForSequenceClassification(args)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.model.to(self.device)

    def load_ckp(self, model, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss

    def save_ckp(self, state, checkpoint_path):
        torch.save(state, checkpoint_path)

    """
    def save_ckp(self, state, is_best, checkpoint_path, best_model_path):
        tmp_checkpoint_path = checkpoint_path
        torch.save(state, tmp_checkpoint_path)
        if is_best:
            tmp_best_model_path = best_model_path
            shutil.copyfile(tmp_checkpoint_path, tmp_best_model_path)
    """

    def train(self):
        total_step = len(self.train_loader) * self.args.train_epochs
        global_step = 0
        eval_step = 100
        best_dev_micro_f1 = 0.0
        for epoch in range(self.args.train_epochs):
            for train_step, train_data in enumerate(self.train_loader):
                self.model.train()
                token_ids = train_data['token_ids'].to(self.device)
                attention_masks = train_data['attention_masks'].to(self.device)
                token_type_ids = train_data['token_type_ids'].to(self.device)
                labels = train_data['labels'].to(self.device)
                train_outputs = self.model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(train_outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print("【train】 epoch：{} step:{}/{} loss：{:.6f}".format(epoch, global_step, total_step, loss.item()))
                global_step += 1
                if global_step % eval_step == 0:
                    dev_loss, dev_outputs, dev_targets = self.dev()
                    accuracy, precision, recall, f1 = self.get_metrics(dev_outputs, dev_targets)
                    print(
                        "【dev】 loss：{:.6f} accuracy：{:.4f} precision：{:.4f} recall：{:.4f} f1：{:.4f}".format(
                            dev_loss, accuracy, precision, recall, f1))
                    if f1 > best_dev_micro_f1:
                        print("------------>保存当前最好的模型")
                        best_dev_micro_f1 = f1
                        checkpoint_path = os.path.join(self.args.output_dir, 'best.pt')
                        torch.save(self.model.state_dict(), checkpoint_path)

    def dev(self):
        self.model.eval()
        total_loss = 0.0
        dev_outputs = []
        dev_targets = []
        with torch.no_grad():
            for dev_step, dev_data in enumerate(self.dev_loader):
                token_ids = dev_data['token_ids'].to(self.device)
                attention_masks = dev_data['attention_masks'].to(self.device)
                token_type_ids = dev_data['token_type_ids'].to(self.device)
                labels = dev_data['labels'].to(self.device)
                outputs = self.model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(),axis=1).flatten()
                dev_outputs.extend(outputs.tolist())
                dev_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, dev_outputs, dev_targets

    def test(self, checkpoint_path):
        model = self.model
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        test_outputs = []
        test_targets = []
        with torch.no_grad():
            for test_step, test_data in enumerate(self.test_loader):
                token_ids = test_data['token_ids'].to(self.device)
                attention_masks = test_data['attention_masks'].to(self.device)
                token_type_ids = test_data['token_type_ids'].to(self.device)
                labels = test_data['labels'].to(self.device)
                outputs = model(token_ids, attention_masks, token_type_ids)
                loss = self.criterion(outputs, labels)
                # val_loss = val_loss + ((1 / (dev_step + 1))) * (loss.item() - val_loss)
                total_loss += loss.item()
                outputs = np.argmax(outputs.cpu().detach().numpy(),axis=1).flatten()
                test_outputs.extend(outputs.tolist())
                test_targets.extend(labels.cpu().detach().numpy().tolist())

        return total_loss, test_outputs, test_targets

    def predict(self, tokenizer, text, args):
        model = self.model
        checkpoint = os.path.join(args.output_dir, 'best.pt')
        model.load_state_dict(torch.load(checkpoint))
        model.eval()
        model.to(self.device)
        with torch.no_grad():
            inputs = tokenizer.encode_plus(text=text,
                                           add_special_tokens=True,
                                           max_length=args.train_max_len,
                                           truncation='longest_first',
                                           padding="max_length",
                                           return_token_type_ids=True,
                                           return_attention_mask=True,
                                           return_tensors='pt')
            token_ids = inputs['input_ids'].to(self.device)
            attention_masks = inputs['attention_mask'].to(self.device)
            token_type_ids = inputs['token_type_ids'].to(self.device)
            outputs = model(token_ids, attention_masks, token_type_ids)
            outputs = np.argmax(outputs.cpu().detach().numpy(),axis=1).flatten().tolist()
            if len(outputs) != 0:
                return outputs[0][0]
            else:
                return '不好意思，我没有识别出来'

    def get_metrics(self, outputs, targets):
        accuracy = accuracy_score(targets, outputs)
        precision = precision_score(targets, outputs, average='micro')
        recall = precision_score(targets, outputs, average='micro')
        micro_f1 = f1_score(targets, outputs, average='micro')
        return accuracy, precision, recall, micro_f1

    def get_classification_report(self, outputs, targets):
        report = classification_report(targets, outputs)
        return report




if __name__ == '__main__':
    config = Config()
    tokenizer = BertTokenizer.from_pretrained(config.bert_dir + 'vocab.txt')
    train_dataset = ClsDataset("./data/train.txt", tokenizer, config.train_max_len)
    dev_dataset = ClsDataset("./data/dev.txt", tokenizer, config.train_max_len)
    test_dataset = ClsDataset("./data/test.txt", tokenizer, config.train_max_len)
    print(train_dataset[0])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    dev_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)


    trainer = Trainer(config, train_loader, dev_loader, test_loader)
    # trainer.train()
    # checkpoint_path = './checkpoints/best.pt'
    # total_loss, test_outputs, test_targets = trainer.test(checkpoint_path)
    # report = trainer.get_classification_report(test_targets, test_outputs)
    # print(report)

    with open(os.path.join('./data/test_my.txt'), 'r') as fp:
        lines = fp.readlines()
        for line in lines:
            line = line.strip().split('\t')
            text = line[0]
            print(text)
            result = trainer.predict(tokenizer, text, config)
            print("预测标签：", result)
            print("真实标签：", line[1])
            print("==========================")
