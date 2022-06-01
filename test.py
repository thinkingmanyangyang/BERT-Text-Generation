# from transformers import load_tf_weights_in_bert, load_tf2_model_in_pytorch_model
# from transformers import BertForPreTraining, BertConfig
# import os
# import tensorflow as tf
# from bert4keras.models import build_transformer_model
#
# config = BertConfig.from_pretrained('user_data/model_data/pretrain_model/chinese_wobert/bert_config.json')
# model = BertForPreTraining(config)
#
# keras_model = build_transformer_model(
#     'user_data/model_data/pretrain_model/chinese_wobert/bert_config.json',
#     'user_data/model_data/pretrain_model/chinese_wobert/bert_model.ckpt',
#     application='unilm',
# )
# keras_model = keras_model.load_weights('user_data/model_data/output_model/fold-1.h5')
# load_tf2_model_in_pytorch_model(model, keras_model)
# model = load_tf_weights_in_bert(model, config, 'user_data/model_data/pretrain_model/tf_weights/bert_model.ckpt')
# model.save_pretrained('user_data/model_data/pretrain_model/pytorch_wobert/')
# os.system('cp user_data/model_data/pretrain_model/chinese_wobert/vocab.txt user_data/model_data/pretrain_model/pytorch_wobert')
# print('convert end...')

import torch
token_type_ids = torch.cat([torch.zeros(3, 3), torch.ones(3, 3), torch.zeros(3,3)], dim=-1)
token_type_ids = token_type_ids.to(dtype=torch.long)

res = torch.cumsum(token_type_ids, dim=1)
b_mask = res[:, None, :] <= res[:, :, None]

batch_size, seq_length = token_type_ids.shape
position_ids = torch.arange(seq_length, dtype=torch.long)
causal_mask = position_ids[None, None, :].repeat(batch_size, seq_length, 1) <= position_ids[None, :, None]
mask = (1 - token_type_ids[:, None, :]) | causal_mask

attention_mask = token_type_ids

print(b_mask.to(attention_mask.dtype) & attention_mask[:, None, :])
