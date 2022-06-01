# coding=utf8
import torch
import os
import time
import math
import json
import jieba
import logging
import argparse
from tqdm import tqdm
from dataset import read_file, \
    get_dataloader, get_test_dataloader
from evaluate import rouge_l
from tokenizer import Tokenizer, load_vocab
from seq2seq_model import Seq2SeqModel
from transformers import BertConfig
from torch.optim import Adam
from utils import set_logger, set_seed
from fgm import FGM
from transformers import AdamW, get_linear_schedule_with_warmup

class Trainer(object):
    def __init__(self, args):
        # 加载数据
        self.args = args
        self.check_path_exist(args)
        pretrain_model_path = args.pretrain_model_path
        recent_model_path = args.recent_model_path  # 用于把已经训练好的模型继续训练
        self.model_save_path = args.model_save_path

        self.device = torch.device(args.device)

        token_dict = load_vocab(
            os.path.join(pretrain_model_path, 'vocab.txt'),
            # simplified=False,
            startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]']
        )
        tokenizer = Tokenizer(
            token_dict=token_dict,
            do_lower_case=True,
            pre_tokenize=lambda s: jieba.cut(s, HMM=False)
        )

        self.tokenizer = tokenizer
        self.train_dataloader, self.eval_dataloader = self.get_dataloader()

        eval_file = os.path.join(self.args.data_dir, 'dev.json')
        dev_examples = read_file(eval_file)
        self.golds = [e.question for e in dev_examples]

        # 从预训练模型加载
        if args.recent_model_path:
            config = BertConfig.from_pretrained(recent_model_path)
            seq2seq_model = Seq2SeqModel.from_pretrained(recent_model_path, config=config,
                                                         tokenizer=tokenizer,
                                                         output_max_length=args.output_max_sequence_length)

        else:
            config = BertConfig.from_pretrained(pretrain_model_path)
            seq2seq_model = Seq2SeqModel.from_pretrained(pretrain_model_path, config=config,
                                                         tokenizer=tokenizer,
                                                         output_max_length=args.output_max_sequence_length)

        seq2seq_model = seq2seq_model.to(self.device)
        print(seq2seq_model)
        logging.info(seq2seq_model)
        self.seq2seq_model = seq2seq_model
        self.optimizer = self.init_optimizer()

        self.best_score = 0.0

    def check_path_exist(self, args):
        # 检查args中的各个路径是否存在，如果不存在则创建
        if not os.path.exists(args.model_save_path):
            os.makedirs(args.model_save_path)

    def get_dataloader(self):
        train_file = os.path.join(args.data_dir, "train.json")
        dev_file = os.path.join(args.data_dir, "dev.json")

        train_dataloader = get_dataloader(train_file, self.tokenizer,
                                               max_length=self.args.max_length,
                                               batch_size=self.args.train_batch_size,
                                               shuffle=True)
        dev_dataloader = get_test_dataloader(dev_file, self.tokenizer,
                                                  max_length=self.args.max_length-self.args.output_max_sequence_length,
                                                  batch_size=self.args.eval_batch_size)
        return train_dataloader, dev_dataloader

    def init_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight', 'transitions']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.seq2seq_model.named_parameters() \
                        if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.seq2seq_model.named_parameters() \
                        if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]
        # t_total = len(self.train_dataloader) // self.args.gradient_accumulation_steps * self.args.epochs
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=1e-8)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_rate * t_total,
        #                                             num_training_steps=t_total)
        return optimizer

    def decode_ids(self, input_ids):
        pred_ids = []
        for idx in input_ids:
            if self.tokenizer.token_to_id(self.tokenizer._token_end) == idx:
                break
            else:
                pred_ids.append(idx)
        return self.tokenizer.decode(pred_ids)

    def eval(self, check=False):
        start_time = time.time()  ## 得到当前时间
        self.seq2seq_model.eval()
        preds, golds = [], []
        for batch in tqdm(self.eval_dataloader, position=0, leave=True):
            inputs = {}
            with torch.no_grad():
                for k, v in batch.items():
                    inputs[k] = v.to(self.device)
                pred = self.seq2seq_model.predict(**inputs)
            import random
            if random.random() < 0.1:
                print(self.decode_ids(pred[0]))
            preds += pred
            if check:
                break
        preds = [self.decode_ids(pred) for pred in preds]
        print(preds[:5])
        logging.info(preds[:5])
        golds = self.golds[:len(preds)]
        score = rouge_l(preds=preds, golds=golds)

        self.seq2seq_model.train()
        end_time = time.time()
        spend_time = end_time - start_time

        report_string = "eval_score = {}, best rouge-l = {}, spend_time = {}".format(
                            score, self.best_score, spend_time)
        promote = ""
        if score > self.best_score:
            promote = " *"
            self.save(self.model_save_path)
            self.best_score = score
        print(report_string + promote)
        logging.info(report_string + promote)
        return score

    def save(self, save_path):
        """
        保存模型
        """
        self.seq2seq_model.save_pretrained(self.model_save_path)
        print("{} saved!".format(save_path))
        logging.info("{} saved!".format(save_path))

    def train(self):
        step = 0
        report_loss = 0
        report_step = 0

        total_step = len(self.train_dataloader)
        eval_step = int(total_step * self.args.val_check_interval)

        self.eval(check=self.args.check)
        self.seq2seq_model.train()

        fgm = FGM(self.seq2seq_model)
        for e in range(self.args.epochs):
            for batch in tqdm(self.train_dataloader, position=0, leave=True):
                step += 1
                # 转移到device上
                inputs = {}
                for k, v in batch.items():
                    inputs[k] = v.to(self.device)

                loss = self.seq2seq_model(**inputs)
                loss.backward()
                report_loss += loss.item()
                report_step += 1
                # # fgm 攻击
                # fgm.attack()
                # loss_adv = self.seq2seq_model(**inputs)
                # loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                # fgm.restore()  # 恢复embedding参数
                # # fgm 攻击 end

                if step % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.seq2seq_model.parameters(), self.args.max_grad_norm)
                    self.optimizer.step()
                    # self.scheduler.step()  # 更新学习率
                    self.seq2seq_model.zero_grad()

                if step % 50 == 0:
                    print('epoch:{}, step:{}, train loss:{}'.format(e, step, report_loss / report_step))
                    logging.info('epoch:{}, step:{}, train loss:{}'.format(e, step, report_loss / report_step))
                    report_step = 0
                    report_loss = 0
                # if step % eval_step == 0 or step == total_step:
            print('epoch:{}, step:{}, train loss:{}'.format(e, step, report_loss / report_step))
            logging.info('epoch:{}, step:{}, train loss:{}'.format(e, step, report_loss / report_step))
            self.eval()

if __name__ == '__main__':
    args_dict = dict(
        data_dir='user_data/tmp_data',
        pretrain_model_path='user_data/model_data/pretrain_model/new_wobert',
        # recent_model_path='user_data/model_data/output_model/seq2seq_model',
        recent_model_path=None,
        model_save_path='user_data/model_data/output_model/seq2seq_model6',
        train_batch_size=8,
        eval_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_rate=0.0,
        max_grad_norm=1.0,
        max_length=400,
        output_max_sequence_length=65,
        learning_rate=1e-5,
        epochs=20,
        weight_decay=0.0,
        val_check_interval=0.5,
        device='cuda:7',
        seed=2021,
        check=True,
    )

    args = argparse.Namespace(**args_dict)
    set_logger('unilm6.log')

    print("***** Running training *****")
    logging.info("***** Running training *****")
    for k, v in args.__dict__.items():
        print("  {:18s} = {}".format(str(k), str(v)))
        logging.info("  {:18s} = {}".format(str(k), str(v)))

    trainer = Trainer(args)
    set_seed(args.seed)

    trainer.train()
    print("this is a test...")
