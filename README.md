# pytorch_simple_bert
更直接的bert代码，可以加载hugging face上的预训练权重，目前支持中文文本分类以及MLM语言模型训练任务。使用的预训练的模型是hugging face上面的hfl/chinese-bert-wwm-ext。训练好的自定义模型（包括分类和基于自己数据继续进行预训练）：阿里云盘链接：https://www.aliyundrive.com/s/SacYb7Q6B3P

# 目录结果
data：存储分类数据<br>
mlm_data：用于生成MLM的数据，这里并没有使用NSP任务，使用的数据是分类数据中的train.txt。<br>
--get_mlm_data.py：用于生成MASK数据。<br>
--mlm_data.txt：MASK数据，每一条是一个句子。<br>
BertEmbeddings.py：获取bert的嵌入。<br>
BertOnlyMLMHead.py：用于进行mlm任务的头部。<br>
draw_loss.py：绘制预训练的损失的文件。<br>
FeedForward.py：前馈网络。<br>
Gelu.py：激活函数。<br>
init_weights.py：初始化模型参数（该文件暂时没有使用）。<br>
loss.png：预训练损失变化图像。<br>
MultiHeadSelfAttention.py：多头自注意力。<br>
Pooler.py：池化操作，用于得到bert的池化输出。<br>
test_BertForMaskedLm.py：自定义bert的预训练代码，可以进行预测。<br>
test_BertForMaskedLMPredict.py：用于测试原始模型的MLM。<br>
test_BertModel.py：用于测试自定义bert模型的输入、输出。<br>
test_BertModelForSequenceClassification.py：自定义bert模型进行分类。<br>
test_pretrained_bert.py：测试原始模型的mlm。	<br>
tokenization.py：从google中bert代码复制而来，修改了读取vocab代码，不需要安装tensorflow。<br>
train.log：训练分类的日志。<br>
train_mlm.log：预训练的日志。<br>
Transformer.py：bert的编码器代码。<br>

# 相关说明
1、我们在自定义bert的时候，为了能够使用预训练的权重，需要对权重进行映射，也就是在load_local2target()函数中进行映射，为了能够适配bert的分类和MLM任务，额外增加了相应的层，比如Pooler层或者BertOnlyMLMHead。在打印自定义bert的权重和原始预训练模型权重的时候，可以发现是对应的了。<br>
2、同样要注意的是多头注意力中要屏蔽掉不是真实字的分数，额外修改了test_gen_attention_masks()以及多头注意力代码中的：
```python
        if attention_mask is not None:
            add_mask = (1.0 - attention_mask.float()) * 1e5
            add_mask = add_mask[:, None, :, :]
            attention_scores -= add_mask
```
3、基于自己数据继续进行预训练任务的时候，以下的代码需要好好看看：
```python
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
```
上述代码的大致意思是：在获取到bert的token级别的输出后，先获取到是进行mask的位置的向量，然后计算对应标签的损失，由于可能有的是进行0填充的（这里是根据生成数据时而定的，比如我们要mask掉6个字，但是在生成时只生成了3个，还有三个需要用0进行填充），我们选择真实的mask的部分，最后对这部分求和，在除以真实长度，最后获得的损失再除以batchsize。

## 分类训练、验证、测试和预测
```python
【train】 epoch：4 step:14061/14065 loss：0.072103
【train】 epoch：4 step:14062/14065 loss：0.040998
【train】 epoch：4 step:14063/14065 loss：0.011044
【train】 epoch：4 step:14064/14065 loss：0.133621

              precision    recall  f1-score   support

           0       0.99      0.99      0.99     17976
           1       1.00      0.99      0.99     18037
           2       0.98      0.99      0.99     17917
           3       1.00      1.00      1.00     17956
           4       0.99      0.99      0.99     17969
           5       1.00      0.99      0.99     18076
           6       0.99      0.99      0.99     18085
           7       1.00      1.00      1.00     17994
           8       0.99      0.99      0.99     18000
           9       1.00      1.00      1.00     17990

    accuracy                           0.99    180000
   macro avg       0.99      0.99      0.99    180000
weighted avg       0.99      0.99      0.99    180000

==========================
国家意志或影响汇市 对美元暂宜持多头思路
预测标签： 0
真实标签： 0
==========================
QQ宝贝零花钱大作战 攒钱拿奖两不误
预测标签： 8
真实标签： 8
==========================
微软：迈向3D游戏还得2-3年时间
预测标签： 8
真实标签： 8
==========================
法国对格鲁吉亚观察员使命未获延期表遗憾
预测标签： 6
真实标签： 6
==========================
媒体报道希拉里出任美国国务卿触发违宪争议
预测标签： 6
真实标签： 6
==========================
朝阳北路低密现房新天际国庆热销备受关注
预测标签： 1
真实标签： 1
==========================
北京四环路以内房价每平方米达17478元
预测标签： 1
真实标签： 1
==========================
13英寸苹果MacBook Pro轻薄时尚本促销
预测标签： 4
真实标签： 4
==========================
瑞星侵权败诉后拒不道歉 法院发公告强制执行
预测标签： 4
真实标签： 4
==========================
图文：安徽发现葫芦形乌龟
预测标签： 5
真实标签： 5
==========================
男子杀死18岁怀孕情人 称怕妻子发现
预测标签： 5
真实标签： 5
==========================
晋级时刻穆帅只留下一个背影 如此国米才配属于他
预测标签： 7
真实标签： 7
==========================
奇才新援10分钟5分5助攻2帽 10天合同竟能捡到宝
预测标签： 7
真实标签： 7
==========================
沪深两市触底反弹 分析人士称中长期仍然看好
预测标签： 2
真实标签： 2
==========================
《经济学人》封面文章：欧洲援助方案
预测标签： 2
真实标签： 2
==========================
```

## 基于自己数据进行预训练
训练结果：
![image](https://github.com/taishan1994/pytorch_simple_bert/blob/main/loss.png)

进行预测：
```python
输入：text = '宇[MASK]员尿液堵塞国际空间站水循环系统'
输出：航
```

# 讲在最后
通过该项目，可以学到以下知识：<br>
1、bert相关组件代码的实现；<br>
2、多头注意力机制中掩码的构建；<br>
3、如何自定义bert，然后导入别人训练好的模型的权重；<br>
4、怎么获取mlm相关的数据；<br>
5、怎么基于自己的数据继续进行预训练；<br>
6、使用自定义的bert进行相关的任务的微调；<br>


# 参考
> https://github.com/whgaara/pytorch-roberta
> https://github.com/brightmart/albert_zh
