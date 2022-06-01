#! -*- coding: utf-8 -*-
# WoBERT做Seq2Seq任务，采用UniLM方案
# 介绍链接：https://kexue.fm/archives/6933
# 数据集：https://github.com/CLUEbenchmark/CLGE 中的LCSTS数据集
# 补充了评测指标bleu、rouge-1、rouge-2、rouge-l
from __future__ import print_function
import json
import numpy as np
from tqdm import tqdm
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
# from bert4keras.optimizers import Adam
from keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from keras.layers import Input
from rouge import Rouge  # pip install rouge
from evaluate import rouge_l
import jieba
import os
jieba.initialize()

# 基本参数
n = 5               # 交叉验证
max_p_len = 194     # 篇章最大长度
max_q_len = 131     # 问题最大长度
max_a_len = 65      # 答案最大长度
head = 64           # 篇章截取中，取答案id前head个字符
batch_size = 8      # 批大小
epochs = 20         # 迭代次数
SEED = 2020         # 随机种子

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
K.clear_session() # 清空当前 session

# 基本参数
# maxlen = 400
batch_size = 8
epochs = 20

# bert配置
config_path = 'user_data/model_data/pretrain_model/chinese_wobert/bert_config.json'
checkpoint_path = 'user_data/model_data/pretrain_model/chinese_wobert/bert_model.ckpt'
dict_path = 'user_data/model_data/pretrain_model/chinese_wobert/vocab.txt'

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    for l in data:
        passage, answer, question = l['passage'], l['answer'], l['question']
        D.append((passage, answer, question))
    return D

# 加载数据集
train_data = load_data('user_data/tmp_data/train.json')
valid_data = load_data('user_data/tmp_data/dev.json')

# 建立分词器
tokenizer = Tokenizer(
    dict_path,
    do_lower_case=True,
    pre_tokenize=lambda s: jieba.cut(s, HMM=False)
)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_o_token_ids = [], [], []
        for is_end, (p, a, q) in self.sample(random):
            p_token_ids, _ = tokenizer.encode(p, maxlen=max_p_len)
            a_token_ids, _ = tokenizer.encode(a, maxlen=max_a_len)
            q_token_ids, _ = tokenizer.encode(q, maxlen=max_q_len)
            token_ids = p_token_ids + a_token_ids[1:] + q_token_ids[1:]
            segment_ids = [0] * (len(p_token_ids) + len(a_token_ids[1:]))
            segment_ids += [1] * (len(token_ids) - len(p_token_ids) - len(a_token_ids[1:]))
            o_token_ids = token_ids
            if np.random.random() > 0.5:
                token_ids = [
                    t if s == 0 or (s == 1 and np.random.random() > 0.3)
                    else np.random.choice(token_ids)
                    for t, s in zip(token_ids, segment_ids)
                ]
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_o_token_ids.append(o_token_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_o_token_ids = sequence_padding(batch_o_token_ids)
                yield [batch_token_ids, batch_segment_ids, batch_o_token_ids], None
                batch_token_ids, batch_segment_ids, batch_o_token_ids = [], [], []



"""构建模型。"""
model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm'
)

o_in = Input(shape=(None,))
train_model = Model(model.inputs + [o_in], model.outputs + [o_in])

# 交叉熵作为loss，并mask掉输入部分的预测
y_true = train_model.input[2][:, 1:]  # 目标tokens
y_mask = train_model.input[1][:, 1:]
y_pred = train_model.output[0][:, :-1]  # 预测tokens，预测与目标错开一位
cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

train_model.add_loss(cross_entropy)
train_model.compile(optimizer=Adam(1e-5))



class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, passage, answer, topk=1):
        passage_ids, _ = tokenizer.encode(passage)
        answer_ids, _ = tokenizer.encode(answer)
        token_ids = passage_ids + answer_ids[1:]
        segment_ids = [0] * len(token_ids)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      1)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=65)

class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.rouge = Rouge()
        self.best_rouge = 0.0

    def on_epoch_end(self, epoch, logs=None):
        metrics = self.evaluate(valid_data)  # 评测模型
        if metrics['rouge-l'] > self.best_rouge:
            self.best_rouge = metrics['rouge-l']
            model.save_weights('./best_model_lcsts.weights')  # 保存模型
        metrics['best_rouge_l'] = self.best_rouge
        print('valid_data:', metrics)

    def evaluate(self, data, topk=1):
        total = 0
        preds = []
        golds = []
        for passage, answer, question in tqdm(data):
            total += 1
            pred_question = ''.join(autotitle.generate(passage, answer, 1))
            preds.append(pred_question)
            golds.append(question)
        print(preds[:5])
        score = rouge_l(preds=preds, golds=golds)
        return {
            # 'rouge-1': rouge_1,
            # 'rouge-2': rouge_2,
            'rouge-l': score,
            # 'bleu': bleu,
        }

if __name__ == '__main__':
    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )

# else:
# data = valid_data
# model.load_weights('./best_model_lcsts.weights')
# preds = []
# golds = []
# for passage, answer, question in tqdm(data):
#     pred_question = ''.join(autotitle.generate(passage, answer, 1))
#     preds.append(pred_question)
#     golds.append(question)
# print(preds[:5])
# score = rouge_l(preds=preds, golds=golds)
# print(score)


