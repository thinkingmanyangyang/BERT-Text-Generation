import json
import re
import argparse
import os

def read_json(file):
    with open(file, encoding='utf-8') as f:
        data = json.load(f)
    print(f'{file} -> data')
    return data

def save_json(data, file):
    with open(file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=1))
    print(f'data -> {file}')

def remove_tags(sequence):
    sequence = sequence.replace(" ", '')
    sequence = sequence.replace("\"", '')
    sequence = sequence.replace("\'", '')
    sequence = sequence.replace("\n", '')
    sequence = sequence.replace("\\", '')
    sequence = sequence.replace('麽', '么')  # 繁转简
    return sequence

def search_idx(str, target):  # 返回target中str的开始下标，若有，则返回list，否则-1
    if str not in target:
        return -1
    idxs = []
    while True:
        if str not in target:
            break
        idxs.append(target.find(str))
        target = target.replace(str, '', 1)
    for i in range(len(idxs)):
        idxs[i] += i * len(str)
    return idxs


def truncate_process_text(process_text, limit_len=50, token=None):
    # 对超过limit_len的answer进行内部截断（保留首尾，截去中间部分）
    # answer在process_text中用token分隔
    idxs = search_idx(token[0], process_text)
    start = idxs[0]+len(token[0])  # 寻找真实答案的起始区间
    end = search_idx(token[1], process_text)[0] -1
    if end-start+1 <= limit_len:
        return process_text
    text_list = list(process_text)
    spicial_tokens = ['，', '：', '。', '；']
    former_idx, later_idx = end, start
    former_patience = 2  # 保留answer前2段
    for i in range(start, end+1):  # 从前往后找
        a = text_list[i]
        if text_list[i] in spicial_tokens:
            if former_patience != 1:
                former_patience -= 1
            else:
                former_idx = i
                break
    for i in range(end-1, start-1, -1):  # 从后往前找
        a = text_list[i]
        if text_list[i] in spicial_tokens:
            later_idx = i
            break
    if former_idx >= later_idx:
        return process_text
    for i in range(former_idx, later_idx+1):
        text_list[i] = ' '
    text_list[former_idx] = '，'
    return ''.join(text_list).replace(' ', '')

def depart_target(content, answer):
    special_token = ['。', '；']
    start_idx = content.find(answer) if content.find(answer) != -1 else content.find(answer[:3])
    former_stop = 0
    end_idx = start_idx + len(answer)
    latter_stop = len(content)-1
    for idx in range(start_idx, -1, -1):
        if content[idx] in special_token:
            former_stop = idx
            break
    for idx in range(end_idx-1, len(content)):
        if content[idx] in special_token:
            latter_stop = idx
            break
    if start_idx == 0 and end_idx == len(content)-1:
        return content[former_stop:latter_stop]
    if start_idx == 0 or former_stop == 0:
        return content[former_stop:latter_stop+1]
    if end_idx == len(content)-1:
        return content[former_stop+1:latter_stop]
    return content[former_stop+1:latter_stop+1]

def truncate(sequence, answer, max_len, id, token=None):
    sequence = remove_tags(sequence)
    answer = remove_tags(answer)
    start_idx = sequence.find(answer) if sequence.find(answer) != -1 else sequence.find(answer[:3])
    end_idx = start_idx + len(answer)
    if start_idx == -1:
        print(f'{id}: {answer}')
        start_idx = end_idx = max_len//2
    sequence_list = list(sequence)
    while(len(sequence_list) > max_len):
        if(start_idx - 0 > len(sequence) - end_idx):
            sequence_list.pop(0)
        else:
            sequence_list.pop(-1)
    res_seq = "".join(sequence_list[:start_idx]) + token[0] + \
              "".join(sequence_list[start_idx:end_idx]) + token[1] + \
              "".join(sequence_list[end_idx:])
    return remove_tags(res_seq)

def split_data(raw_data, vaild_rate=0.1):
    train_data = raw_data[:int(len(raw_data) * (1 - vaild_rate))]
    vaild_data = raw_data[int(len(raw_data) * (1 - vaild_rate)):]
    return train_data, vaild_data
    parent_dir = os.path.dirname(data_file)
    save_json(train_data, os.path.join(parent_dir, 'split_train.json'))
    save_json(vaild_data, os.path.join(parent_dir, 'split_dev.json'))

def transform_format(input_data, isProcess_text, isTruncate_process_text, isAdd_raw_text=False, token=None):
    output_data = []
    for data in input_data:
        content = data['text']
        annotations = data['annotations']
        for a in annotations:
            question = a['Q'].strip()
            answer = a['A'].strip()
            answer_list = list(answer)
            answer_list[0] = '' if answer_list[0] == '。' else answer_list[0]
            answer = ''.join(answer_list)
            new_content = truncate(content, answer, 450, data['id'], token)  # 保留完整context
            old_content = new_content
            if isProcess_text:  # 抽取片段
                new_content = truncate(depart_target(content, answer), answer, 450, data['id'], token)
                if isTruncate_process_text:  # 对抽取的片段进行answer内截断
                    new_content = truncate_process_text(new_content, limit_len=50, token=token)
            if (len(new_content) + len(question)) > 512:
                print(len(new_content) + len(question))
            output_data.append(
                {"src": new_content,
                 "tgt": question,
                 })

            if isAdd_raw_text:
                output_data.append(
                    {
                        'src': old_content,
                        'tgt': question
                    }
                )
    return output_data

def sort_data(data):
    return sorted(data, key=lambda example: len(example['tgt']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_name", default="data", type=str, required=False,
                        help="输入数据的目录")
    parser.add_argument("--output_file_name", default="user_data/tmp_data", type=str, required=False,
                        help="输出数据的目录")
    parser.add_argument("--process_text", default=True, type=bool, required=False,
                        help='是否抽取答案构成简短输入')
    parser.add_argument('--truncate_process_text', action='store_true', required=False,
                        help='是否对process_text中的answer进行内部截断')
    parser.add_argument('--add_raw_text', default=False, type=bool, required=False,
                        help='是否在训练数据中将原始数据也作为训练语料')
    parser.add_argument('--depart_type', default=0, type=int, required=False)

    args = parser.parse_args()
    print("是否对process_text中的answer进行内部截断", args.truncate_process_text)
    depart_token_list = [['[SEP]', '[SEP]'], ['[#]', '[$]'], ['(', ')'], ['<S>', '<T>']]
    depart_token = depart_token_list[args.depart_type]  # '将answer与content分隔的标识符'
    tcm = True  # 设置加载的数据
    if tcm:
        input_data = read_json(os.path.join(args.input_file_name, 'round1_train_0907.json'))
        test_data = read_json(os.path.join(args.input_file_name, 'juesai.json'))
        save_json(test_data, os.path.join(args.output_file_name, 'raw_test.json'))

        train_data, val_data = split_data(input_data)

        save_json(val_data, os.path.join(args.output_file_name, 'raw_dev.json'))
        train_data = transform_format(train_data, args.process_text, args.truncate_process_text, args.add_raw_text, token=depart_token)
        val_data = transform_format(val_data, args.process_text, args.truncate_process_text, token=depart_token)
        test_data = transform_format(test_data, args.process_text, args.truncate_process_text, token=depart_token)

        # train_data = train_data[:20]
        # val_data = train_data[:5]
        val_data = sort_data(val_data)

        print('data lengths', len(train_data), len(val_data), len(test_data))
        save_json(train_data, os.path.join(args.output_file_name, 'train.json'))
        save_json(val_data, os.path.join(args.output_file_name, 'dev.json'))
        save_json(test_data, os.path.join(args.output_file_name, 'test.json'))
    else:  # dureader: 400, 25
        input_data = read_json(os.path.join(args.input_file_name, 'dureader_train.json'))
        test_data = read_json(os.path.join(args.input_file_name, 'dureader_dev.json'))
        raw_dev = read_json(os.path.join(args.input_file_name, 'tcm_dureader_dev.json'))

        save_json(raw_dev, os.path.join(args.output_file_name, 'raw_dev.json'))
        save_json(raw_dev, os.path.join(args.output_file_name, 'raw_test.json'))
        save_json(input_data, os.path.join(args.output_file_name, 'train.json'))
        save_json(test_data, os.path.join(args.output_file_name, 'dev.json'))
        save_json(test_data, os.path.join(args.output_file_name, 'test.json'))

    print("ending...")

