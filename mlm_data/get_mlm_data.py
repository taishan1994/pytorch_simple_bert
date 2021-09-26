import sys
sys.path.append('..')
import re
import random
import json
import tokenization
import jieba
from tqdm import tqdm


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels, ):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def get_new_segment(segment):
    """
   输入一句话，返回一句经过处理的话: 为了支持中文全称mask，将被分开的词，将上特殊标记("#")，使得后续处理模块，能够知道哪些字是属于同一个词的。
   :param segment: 一句话. e.g.  ['悬', '灸', '技', '术', '培', '训', '专', '家', '教', '你', '艾', '灸', '降', '血', '糖', '，', '为', '爸', '妈', '收', '好', '了', '！']
   :return: 一句处理过的话 e.g.    ['悬', '##灸', '技', '术', '培', '训', '专', '##家', '教', '你', '艾', '##灸', '降', '##血', '##糖', '，', '为', '爸', '##妈', '收', '##好', '了', '！']
   """
    seq_cws = jieba.lcut("".join(segment))  # 分词
    # print(seq_cws)
    seq_cws_dict = {x: 1 for x in seq_cws}  # 分词后的词加入到词典dict
    new_segment = []
    i = 0
    while i < len(segment):  # 从句子的第一个字开始处理，知道处理完整个句子
        if len(re.findall('[\u4E00-\u9FA5]', segment[i])) == 0:  # 如果找不到中文的，原文加进去即不用特殊处理。
            new_segment.append(segment[i])
            i += 1
            continue

        has_add = False
        for length in range(3, 0, -1):
            if i + length > len(segment):
                continue
            if ''.join(segment[i:i + length]) in seq_cws_dict:
                new_segment.append(segment[i])
                for l in range(1, length):
                    new_segment.append('##' + segment[i + l])
                i += length
                has_add = True
                break
        if not has_add:
            new_segment.append(segment[i])
            i += 1
        # print("get_new_segment.wwm.get_new_segment:",new_segment)
    # print(new_segment)
    return new_segment


def create_masked_lm_predictions(tokens, vocab_words, args):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        if (args.do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])
    # print(cand_indexes)
    args.rng.shuffle(cand_indexes)
    if args.non_chinese == False:  # if non chinese is False, that means it is chinese, then try to remove "##" which is added previously
        output_tokens = [t[2:] if len(re.findall('##[\u4E00-\u9FA5]', t)) > 0 else t for t in tokens]  # 去掉"##"
    else:  # english and other language, which is not chinese
        output_tokens = list(tokens)
    num_to_predict = args.max_predictions_per_seq
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        for index in index_set:
            covered_indexes.add(index)
            masked_token = None
            if args.rng.random() < 0.6:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if args.rng.random() < 0.5:
                    if args.non_chinese == False:  # if non chinese is False, that means it is chinese, then try to remove "##" which is added previously
                        masked_token = tokens[index][2:] if len(re.findall('##[\u4E00-\u9FA5]', tokens[index])) > 0 else \
                        tokens[index]  # 去掉"##"
                    else:
                        masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    # （672, 7992）
                    masked_token = vocab_words[args.rng.randint(0, len(vocab_words) - 1)]
            output_tokens[index] = masked_token
            masked_lms.append((index, tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x[0])
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p[0])
        masked_lm_labels.append(p[1])
    # print(output_tokens, masked_lm_positions, masked_lm_labels)
    return (output_tokens, masked_lm_positions, masked_lm_labels)


def create_instances_from_line_by_line(args):
    tokenizer = tokenization.FullTokenizer(
        vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    # sen = "巴黎是法国的首都，happy！longly"
    # print(tokenizer.tokenize(sen))
    # 将多个空格替换成一个空格
    with open(args.vocab_file, 'r') as fp:
        vocabs = fp.read().strip().split('\n')
    vocab_words = {i: word for i, word in enumerate(vocabs)}
    instances = []
    with open(args.train_file, 'r') as fp:
        lines = fp.read().strip().split('\n')
        for i,line in enumerate(lines):
            line = line.split('\t')
            text = line[0]
            # 将多个空格替换为一个空格
            text = re.sub("[ ]+", ' ', text).strip()
            # 将字符转换为unicode编码
            text = tokenization.convert_to_unicode(text)
            # print(text)
            # 对句子进行token化
            tokens = tokenizer.tokenize(text)
            # print(tokens)
            segment = get_new_segment(tokens)
            # 最大长度要减去[CLS]和[SEP]，这里预先就规定最大长度，后续获得Dataset的时候就不用考虑了
            segment = segment[:args.max_seq_length-2]
            # print(segment)
            token_word = []
            segment_ids = []
            token_word.append("[CLS]")
            segment_ids.append(0)
            for token in segment:
                token_word.append(token)
                segment_ids.append(0)
            token_word.append("[SEP]")
            segment_ids.append(0)
            # print(token_char)
            # print(segment_ids)
            (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(token_word, vocab_words,
                                                                                           args)
            instance = TrainingInstance(  # 创建训练实例的对象
                tokens=tokens,
                segment_ids=segment_ids,
                masked_lm_positions=masked_lm_positions,
                masked_lm_labels=masked_lm_labels)
            instances.append(instance)
            if i < 3:
                print(instance)
    return instances

def write_instance_to_example_files(instances, args):
    with open(args.output_file,'w',encoding='utf-8') as fp:
        total = len(instances)
        print('总共有数据：{}条'.format(len(instances)))
        for instance in tqdm(instances):
            raw_example = {}
            raw_example['tokens'] = instance.tokens
            raw_example['segment_ids'] = instance.segment_ids
            raw_example['masked_lm_positions'] = instance.masked_lm_positions
            raw_example['masked_lm_labels'] = instance.masked_lm_labels
            fp.write(json.dumps(raw_example, ensure_ascii=False) + '\n')

def main(args):
    instances = create_instances_from_line_by_line(args)
    write_instance_to_example_files(instances, args)

class Args:
    vocab_file = '../../../model_hub/hfl_chinese-bert-wwm-ext/vocab.txt'
    train_file = '../data/train.txt'
    output_file = './mlm_data.txt'
    seed = 123456
    do_lower_case = True
    max_seq_length = 64
    do_whole_word_mask = True
    masked_lm_prob = 0.10  # 替换为[MASK]的比例
    # int(64*0.10) = 6
    max_predictions_per_seq = int(max_seq_length * masked_lm_prob)  # 一个句子中进行预测的token比例
    rng = random.Random(seed)
    non_chinese = False


if __name__ == '__main__':
    args = Args()
    main(args)
