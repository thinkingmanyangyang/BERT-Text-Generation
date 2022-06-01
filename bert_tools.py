# bert tools
# 功能: 实现词表精简 添加
# author: manyangyang
import os
import json
import torch
import logging
from torch import nn
import math


def swish(x):
    return x * torch.sigmoid(x)


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))


ACT2FN = {"gelu":torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.answer_tag_embeddings = nn.Embedding(3, config.hidden_size)
        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()


class UnilmPretrainedModel(nn.Module):
    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                )
            )
        # Save config in model
        self.config = config
        self.base_model_prefix = 'bert.'

    # init weights always no use, cause we load weight from pre-trained model
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def apply(self, fn):
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    def init_weights(self):
        self.apply(self._init_weights)

    @classmethod
    def load_weight_form_pretrained(cls, pretrain_path, config, *model_args, **model_kwargs):
        model_weight_dir = os.path.join(pretrain_path, 'pytorch_model.bin')
        state_dict = torch.load(model_weight_dir, map_location="cpu")
        keep_tokens = model_kwargs.pop("keep_tokens", None)
        token_dict = model_kwargs.pop("token_dict", None)
        for key in state_dict.keys():
            if 'word_embeddings' in key or 'predictions.decoder' in key or 'predictions.bias' in key:
                print(key, state_dict[key].shape)
                word_embeddings = state_dict[key]
                if keep_tokens is not None:
                    word_embeddings = cls.simply_embeddings(word_embeddings, keep_token=keep_tokens)
                    config.vocab_size = len(keep_tokens)
                if token_dict is not None:
                    word_embeddings = cls.add_new_tokens(word_embeddings, new_token_dict=token_dict, std=config.initializer_range)
                    config.vocab_size = len(token_dict)
                state_dict[key] = word_embeddings

        model = cls(config, *model_args, **model_kwargs)
        model.load_state_dict(state_dict, prefix=model.base_model_prefix)
        state_dict[key] = word_embeddings
        return model

    def load_state_dict(self, state_dict, prefix='', strict=True):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    # print(prefix+name+'.')
                    load(child, prefix + name + '.')

        load(self, prefix=prefix)
        load = None  # break load->load reference cycle

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(
                    0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(
                    0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        print(error_msgs)
        logging.info(error_msgs)
        return missing_keys, unexpected_keys, error_msgs

    @staticmethod
    def simply_embeddings(word_embeddings, keep_token):
        word_embeddings = word_embeddings[keep_token]
        return word_embeddings

    @classmethod
    def add_new_tokens(cls, word_embeddings, new_token_dict, std=0.02):
        if len(new_token_dict) <= word_embeddings.shape[0]:
            return word_embeddings

        new_vocab_size = len(new_token_dict)
        if len(word_embeddings.shape) == 2:
            vocab_size, embedding_dim = word_embeddings.shape
            new_embeddings = torch.normal(mean=torch.zeros(new_vocab_size, embedding_dim), std=std)
            new_embeddings[:vocab_size, :] = word_embeddings
            return new_embeddings
        elif len(word_embeddings.shape) == 1:
            vocab_size = word_embeddings.shape[0]
            new_bias = torch.zeros(new_vocab_size)
            new_bias[:vocab_size] = word_embeddings
            return new_bias

    def save_config_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        config_save_path = os.path.join(save_path, 'config.json')
        model_save_path = os.path.join(save_path, 'pytorch_model.bin')

        # Only save the model itself if we are using distributed training
        model_to_save = self.module if hasattr(self, "module") else self

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        with open(config_save_path, 'w', encoding='utf8') as f:
            json.dump(model_to_save.config.__dict__, f, indent=4)

        torch.save(model_to_save.state_dict(), model_save_path)

class BertModel(UnilmPretrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()


class BertForPreTraining(UnilmPretrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()
        self.base_model_prefix = ''


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)


class BertConfig(object):
    model_type = "bert"
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        for k, v in kwargs.items():
            setattr(self, k, v)


    @classmethod
    def load_config(cls, pretrain_path):
        config_dir = os.path.join(pretrain_path, 'config.json')
        with open(config_dir, 'r', encoding='utf8') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


if __name__ == '__main__':
    config = BertConfig.load_config('pytorch_wobert')

    model = BertForPreTraining.load_weight_form_pretrained('pytorch_wobert', config=config, keep_tokens=[0], token_dict=[1,2,3,34,45])
    model.save_config_model('new_wobert')
    # state_dict = torch.load('pytorch_wobert/pytorch_model.bin')
    # config = BertConfig()
    # model = BertForPreTraining()
    # BertForPreTraining.from_pretrained()
    # BertForPreTraining.save_pretrained()
    # model.load_state_dict()


