# from rouge import rouge_l_sentence_level
# # pip install easy-rouge
from rouge import Rouge
import json
import argparse

rouge = Rouge()

def rouge_l(preds, golds):
    assert len(preds) == len(golds)
    score = 0
    for pred, gold in zip(preds, golds):
        pred = ' '.join(pred)
        gold = ' '.join(gold)
        if pred.strip():
            scores = rouge.get_scores(hyps=pred, refs=gold)
            score += scores[0]['rouge-l']['f']
    return score / len(golds)

if __name__ == '__main__':
    res = rouge_l(['我今天x键盘不好用'], ['今天键盘x不好用XX'])
    print(res)
