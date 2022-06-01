import json
import re
import os
from collections import defaultdict
import pandas as pd

max_p_len = 194     # 篇章最大长度
max_q_len = 131     # 问题最大长度
max_a_len = 65      # 答案最大长度
head = 64           # 篇章截取中，取答案id前head个字符

def json2df(filename):
    """json转pandas.DataFrame。"""
    json_data = json.load(open(filename, encoding='utf8'))
    D = defaultdict(list)
    for d in json_data:
        for qa in d['annotations']:
            D['passage'].append(d['text'])
            D['question'].append(qa['Q'])
            D['answer'].append(qa['A'])

    return pd.DataFrame(D)


def preprocess(df):
    """数据预处理。"""
    # 剔除空白字符
    df = df.applymap(lambda x: re.sub(r'\s', '', x))
    df = df.applymap(lambda x: re.sub(r'\\n', '', x))

    # 剔除带括号的英文
    func = lambda m: '' if len(m.group(0)) > 5 else m.group(0)
    df = df.applymap(lambda x: re.sub(r'\([A-Za-z]+\)', func, x))
    df = df.applymap(lambda x: re.sub(r'（[A-Za-z]+）', func, x))

    # 筛选出答案与篇章不匹配的数据
    tmp = list()
    for idx, row in df.iterrows():
        if row['answer'] not in row['passage']:
            tmp.append(idx)

    # 处理部分不匹配数据
    no_match = df.loc[tmp]
    df.drop(index=tmp, inplace=True)
    no_match['answer'] = no_match['answer'].map(lambda x: x.replace('.', ''))
    df = pd.concat([df, no_match])
    df.reset_index(drop=True, inplace=True)

    return df

def load_data(filename):
    """加载数据。"""
    df = json2df(filename)  # json转DataFrame
    df = preprocess(df)     # 数据预处理

    # 文本截断
    D = list()
    for _, row in df.iterrows():
        passage = row['passage'].strip()
        question = row['question'].strip()
        answer = row['answer'].strip()
        if len(passage) < max_p_len - 2 and len(answer) < max_a_len - 1:
            D.append((passage, question, answer))
        else:
            a = answer[:max_a_len-1] if len(answer) > max_a_len - 1 else answer
            try:
                idx = passage.index(a)
                if len(passage[idx:]) < (max_p_len - 2 - head):
                    p = passage[-(max_p_len - 2):]
                else:
                    p = passage[max(0, idx - head):]
                    p = p[:max_p_len - 2]
            except ValueError:
                p = passage[:max_p_len - 2]
            D.append((p, question, a))
    return D

def split_data(raw_data, vaild_rate=0.2):
    train_data = raw_data[:int(len(raw_data) * (1 - vaild_rate))]
    vaild_data = raw_data[int(len(raw_data) * (1 - vaild_rate)):]
    return train_data, vaild_data


def transform_format(data):
    format_data = []
    for d in data:
        p, q, a = d
        format_data.append({
            'src': p + '[SEP]' + a + '[SEP]',
            'tgt': q,
        })
    return format_data


def save_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=1))
    print(f'data -> {file}')

def sort_data(data):
    return sorted(data, key=lambda example: len(example['tgt']))

if __name__ == '__main__':
    data = load_data('round1_train_0907.json')
    train_data, val_data = split_data(data)
    train_data = transform_format(train_data)
    val_data = transform_format(val_data)
    val_data = sort_data(val_data)
    save_json(train_data, '../user_data/tmp_data/train.json')
    save_json(val_data, '../user_data/tmp_data/dev.json')






