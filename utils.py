import os
import json
import logging
import random
import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification

LABEL_LIST = ['angry', 'surprise', 'fear', 'happy', 'sad', 'neural']
SMP_2019_LABEL_LIST = ['0', '1', '2']

def remove_tags(sequence):
    sequence = sequence.replace(" ", '')
    sequence = sequence.replace("\"", '')
    sequence = sequence.replace("\'", '')
    sequence = sequence.replace("\n", '')
    sequence = sequence.replace("\\", '')
    sequence = sequence.strip()
    return sequence

def read_json(file):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)
    print(f'{file} -> data')
    return data

def save_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=4))
    print(f'data -> {file}')

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)


    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    # # Logging to console
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(logging.Formatter('%(message)s'))
    # logger.addHandler(stream_handler)


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty=1):
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty # 长度惩罚的指数系数
        self.num_beams = num_beams # beam size
        self.beams = [] # 存储最优序列及其累加的log_prob score
        self.worst_score = 1e9 # 将worst_score初始为无穷大。

    def __len__(self):
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        score = sum_logprobs / (len(hyp) + 1e-8) ** self.length_penalty # 计算惩罚后的score
        if len(self) < self.num_beams or score > self.worst_score:
            # 如果类没装满num_beams个序列
            # 或者装满以后，但是待加入序列的score值大于类中的最小值
            # 则将该序列更新进类中，并淘汰之前类中最差的序列
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                # 如果没满的话，仅更新worst_score
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        # 当解码到某一层后, 该层每个结点的分数表示从根节点到这里的log_prob之和
        # 此时取最高的log_prob, 如果此时候选序列的最高分都比类中最低分还要低的话
        # 那就没必要继续解码下去了。此时完成对该句子的解码，类中有num_beams个最优序列。
        if len(self) < self.num_beams:
            return False
        else:
            cur_score = best_sum_logprobs / (cur_len + 1e-8) ** self.length_penalty
            # cur_score = best_sum_logprobs / cur_len
            ret = self.worst_score >= cur_score
            return ret
