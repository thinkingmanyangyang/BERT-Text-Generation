from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch
import json
from tqdm import tqdm
import numpy as np

class Seq2SeqInputExample(object):
    def __init__(self, passage, answer, question):
        super(Seq2SeqInputExample, self).__init__()
        self.passage = passage
        self.answer = answer
        self.question = question


class Seq2SeqInputFeature(object):
    def __init__(self, input_ids, token_type_ids, position_ids, attention_mask, answer_tag_ids=None):
        super().__init__()
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.position_ids = position_ids
        self.attention_mask = attention_mask
        self.answer_tag_ids = answer_tag_ids


class Seq2SeqDataset(Dataset):
    """
    针对特定数据集，定义一个相关的取数据的方式
    """
    def __init__(self, features):
        # 一般init函数是加载所有数据
        super(Seq2SeqDataset, self).__init__()
        self.features = features

    def __getitem__(self, i):
        # 得到单个数据
        return self.features[i]

    def __len__(self):
        return len(self.features)


def read_file(input_dir):
    with open(input_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
    examples = []
    for part in data:
        examples.append(
            Seq2SeqInputExample(
                passage=part['passage'],
                answer=part['answer'],
                question=part['question'],
            )
        )
    return examples


def seq2seq_convert_example_to_feature(examples, tokenizer, max_length, is_test=False):
    features = []
    print("max length = ", max_length)
    print('example nums = ', len(examples))
    for example in tqdm(examples, desc='convert to features', leave=True):
        # print(example.question)
        # if is_test:
        #     input_ids, token_type_ids = \
        #         tokenizer.encode(example.passage+'#'+example.answer,
        #                          maxlen=max_length)
        # else:
        #     input_ids, token_type_ids = \
        #         tokenizer.encode(example.passage+'#'+example.answer,
        #                          example.question,
        #                          maxlen=max_length,
        #                         )
        if is_test:
            input_ids, token_type_ids, answer_tag_ids = \
                tokenizer.encode_plus(passage=example.passage,
                                      answer=example.answer,
                                      maxlen=max_length)
        else:
            input_ids, token_type_ids, answer_tag_ids = \
                tokenizer.encode_plus(passage=example.passage,
                                      answer=example.answer,
                                      question=example.question,
                                      maxlen=max_length,
                                      )

        position_ids = [i for i in range(len(input_ids))]
        attention_mask = [1 for i in range(len(input_ids))]
        features.append(
            Seq2SeqInputFeature(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                answer_tag_ids=answer_tag_ids,
            )
        )
    return features


def collate_batch(features):
    features = merge_dict(features)
    max_len = max([len(f) for f in features['input_ids']])
    batch = {}
    for k, v in features.items():
        if v is not None and not isinstance(v, str):
            values = padding(k, v, max_len)
            batch[k] = torch.tensor(values, dtype=torch.long)
    return batch


def merge_dict(features):
    first = features[0]
    batch = {}
    for k, v in vars(first).items():
        if v is not None:
            values = [getattr(f, k) for f in features]
        else:
            values = None
        batch[k] = values
    return batch


def padding(feature_name, features, max_len):
    values = list()
    for f in features:
        pad_len = max_len - len(f)
        if feature_name != 'position_ids':
            pad_part = [0] * pad_len
        else:
            pad_part = f[-1:] * pad_len
        values.append(f + pad_part)
    return values


def get_dataloader(input_dir,
                   tokenizer,
                   max_length=512,
                   batch_size=1,
                   shuffle=True):
    train_dataset = Seq2SeqDataset(
                seq2seq_convert_example_to_feature(
                    examples=read_file(input_dir),
                    tokenizer=tokenizer,
                    max_length=max_length,
                )
            )
    train_sampler = RandomSampler(train_dataset)
    return \
        DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=collate_batch,
        )


def get_test_dataloader(input_dir,
                        tokenizer,
                        max_length=400,
                        batch_size=1):
    test_dataloader = Seq2SeqDataset(
        seq2seq_convert_example_to_feature(
            examples=read_file(input_dir),
            tokenizer=tokenizer,
            max_length=max_length,
            is_test=True,
        ),
    )
    sequential_sampler = SequentialSampler(test_dataloader)
    return DataLoader(
        test_dataloader,
        batch_size=batch_size,
        sampler=sequential_sampler,
        collate_fn=collate_batch,
    )


