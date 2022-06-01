from tokenizer import load_vocab, Tokenizer
import jieba
import json
import os

pretrain_path = 'user_data/model_data/pretrain_model/pytorch_wobert'
train_vocab = set()
with open('data/round1_train_0907.json', 'r', encoding='utf8') as f:
    data = json.load(f)
    for item in data:
        passage = item['text']
        train_vocab.update(jieba.cut(passage))
        train_vocab.update(list(passage))
        annotations = item['annotations']
        for annotation in annotations:
            question = annotation['Q']
            answer = annotation['A']
            train_vocab.update(jieba.cut(question))
            train_vocab.update(list(question))
            train_vocab.update(jieba.cut(answer))
            train_vocab.update(list(answer))
special_token = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
train_vocab = special_token + list(set(train_vocab))
print(len(train_vocab))
def save_vocab(file_name, vocab):
    with open(file_name, 'w', encoding='utf8') as f:
        for line in vocab:
            f.write(line + '\n')
save_vocab('data/train_vocab.txt', train_vocab)

wobert_vocab = load_vocab(os.path.join(pretrain_path, 'vocab.txt'), simplified=False)
train_vocab = load_vocab('data/train_vocab.txt', simplified=False)

def char_in_vocab(word, token_dict):
    for char in word:
        if char not in token_dict:
            return False
    return True

def update_vocab(source_vocab, target_vocab):
    for word, _ in target_vocab.items():
        if word not in source_vocab:
            for char in word:
                if char not in source_vocab:
                    source_vocab[char] = len(source_vocab)
    return source_vocab

def simply_vocab(source_vocab, target_vocab):
    new_token_dict, keep_tokens = {}, []
    for word, id in source_vocab.items():
        if word in target_vocab:
            new_token_dict[word] = len(new_token_dict)
            keep_tokens.append(id)
    return new_token_dict, keep_tokens


new_token_dict, keep_tokens = simply_vocab(wobert_vocab, train_vocab)
new_token_dict = update_vocab(new_token_dict, train_vocab)
print('new_vocab_size:', len(new_token_dict))
from bert_tools import BertForPreTraining, BertConfig
pretrain_path = 'user_data/model_data/pretrain_model/pytorch_wobert'
config = BertConfig.load_config(pretrain_path=pretrain_path)
model = BertForPreTraining.load_weight_form_pretrained(pretrain_path=pretrain_path,
                                                       config=config,
                                                       keep_tokens=keep_tokens,
                                                       token_dict=new_token_dict
                                                       )
model.save_config_model('user_data/model_data/pretrain_model/new_wobert')
save_vocab('user_data/model_data/pretrain_model/new_wobert/vocab.txt', new_token_dict.keys())




