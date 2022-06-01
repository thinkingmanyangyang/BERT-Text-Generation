import keras.backend.tensorflow_backend as KTF
import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K, search_layer
from bert4keras.models import build_transformer_model
# from bert4keras.optimizers import Adam
from keras.optimizers import Adam
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from bert4keras.snippets import sequence_padding
from bert4keras.tokenizers import Tokenizer, load_vocab
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.models import Model
from rouge import Rouge  # pip install rouge
from sklearn.model_selection import KFold
from tqdm import tqdm
import json

# from utils import json2df, preprocess

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# 基本参数
n = 5               # 交叉验证
max_p_len = 194     # 篇章最大长度
max_q_len = 131     # 问题最大长度
max_a_len = 65      # 答案最大长度
head = 64           # 篇章截取中，取答案id前head个字符
batch_size = 8      # 批大小
epochs = 20         # 迭代次数
SEED = 2020         # 随机种子

# nezha配置
config_path = '../user_data/model_data/pretrain_model/chinese_wobert/bert_config.json'
checkpoint_path = '../user_data/model_data/pretrain_model/chinese_wobert/bert_model.ckpt'
dict_path = '../user_data/model_data/pretrain_model/chinese_wobert/vocab.txt'

# 加载并精简词表，建立分词器
token_dict = load_vocab(
    dict_path=dict_path,
    simplified=False,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


def load_data(filename):
    """加载数据。"""
    D = []
    with open(filename, encoding='utf-8') as f:
        data = json.load(f)
    for l in data:
        passage, answer, question = l['passage'], l['answer'], l['question']
        D.append((passage, answer, question))
    return D


class data_generator(DataGenerator):
    """数据生成器。"""
    def __init__(self, data, batch_size=32, buffer_size=None, random=False):
        super().__init__(data, batch_size, buffer_size)
        self.random = random

    def __iter__(self, random=False):
        """单条样本格式：[CLS]篇章[SEP]答案[SEP]问题[SEP]。"""
        batch_token_ids, batch_segment_ids, batch_o_token_ids = [], [], []
        for is_end, (p, q, a) in self.sample(random):
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


def build_model():
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

    return model, train_model

class QuestionGeneration(AutoRegressiveDecoder):
    """通过beam search来生成问题。"""
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, passage, answer, topk=1):
        p_token_ids, _ = tokenizer.encode(passage, maxlen=max_p_len)
        a_token_ids, _ = tokenizer.encode(answer, maxlen=max_a_len)
        token_ids = p_token_ids + a_token_ids[1:]
        segment_ids = [0] * (len(p_token_ids) + len(a_token_ids[1:]))
        q_ids = self.beam_search([token_ids, segment_ids], 1)  # 基于beam search
        return tokenizer.decode(q_ids)


class Evaluator(keras.callbacks.Callback):
    """计算验证集rouge_l。"""
    def __init__(self, valid_data, qg):
        super().__init__()
        self.rouge = Rouge()
        self.best_rouge_l = 0.
        self.valid_data = valid_data
        self.qg = qg

    def on_epoch_end(self, epoch, logs=None):
        rouge_l = self.evaluate(self.valid_data)  # 评测模型
        if rouge_l > self.best_rouge_l:
            self.best_rouge_l = rouge_l
        logs['val_rouge_l'] = rouge_l
        print(
            f'val_rouge_l: {rouge_l:.5f}, '
            f'best_val_rouge_l: {self.best_rouge_l:.5f}',
            end=''
        )

    def evaluate(self, data, topk=1):
        total, rouge_l = 0, 0
        for p, q, a in tqdm(data):
            total += 1
            q = ' '.join(q)
            pred_q = ' '.join(self.qg.generate(p, a, 1))
            if pred_q.strip():
                scores = self.rouge.get_scores(hyps=pred_q, refs=q)
                rouge_l += scores[0]['rouge-l']['f']
        rouge_l /= total
        return rouge_l

def split_data(raw_data, vaild_rate=0.2):
    train_data = raw_data[:int(len(raw_data) * (1 - vaild_rate))]
    vaild_data = raw_data[int(len(raw_data) * (1 - vaild_rate)):]
    return train_data, vaild_data

def do_train():
    data = load_data('round1_train_0907.json')  # 加载数据

    # 交叉验证
    kf = KFold(n_splits=n, shuffle=True, random_state=SEED)
    for fold, (trn_idx, val_idx) in enumerate(kf.split(data), 1):
        print(f'Fold {fold}')

        # 配置Tensorflow Session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
        sess = tf.Session(config=config)
        KTF.set_session(sess)

        # 划分训练集和验证集
        train_data = [data[i] for i in trn_idx]
        valid_data = [data[i] for i in val_idx]

        train_generator = data_generator(train_data, batch_size, random=True)

        model, train_model = build_model()  # 构建模型

        # 问题生成器
        qg = QuestionGeneration(
            model, start_id=None, end_id=tokenizer._token_dict['？'],
            maxlen=max_q_len
        )

        # 设置回调函数
        callbacks = [
            Evaluator(valid_data, qg),
            EarlyStopping(
                monitor='val_rouge_l',
                patience=1,
                verbose=1,
                mode='max'),
            ModelCheckpoint(
                f'../user_data/model_data/fold-{fold}.h5',
                monitor='val_rouge_l',
                save_weights_only=False,
                save_best_only=True,
                verbose=1,
                mode='max'),
        ]

        # 模型训练
        train_model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=callbacks,
        )

        KTF.clear_session()
        sess.close()

def new_train():

    # 配置Tensorflow Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    sess = tf.Session(config=config)
    KTF.set_session(sess)

    train_data = load_data('../user_data/tmp_data/train.json')
    valid_data = load_data('../user_data/tmp_data/dev.json')

    train_generator = data_generator(train_data, batch_size, random=True)
    model, train_model = build_model()  # 构建模型

    # 问题生成器
    qg = QuestionGeneration(
        model, start_id=None, end_id=tokenizer._token_dict['？'],
        maxlen=max_q_len
    )

    # 设置回调函数
    callbacks = [
        Evaluator(valid_data, qg),
        EarlyStopping(
            monitor='val_rouge_l',
            patience=1,
            verbose=1,
            mode='max'),
        ModelCheckpoint(
            f'../user_data/model_data/keras-model.h5',
            monitor='val_rouge_l',
            save_weights_only=False,
            save_best_only=True,
            verbose=1,
            mode='max'),
    ]

    # 模型训练
    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=callbacks,
    )

    KTF.clear_session()
    sess.close()
if __name__ == '__main__':
    new_train()
