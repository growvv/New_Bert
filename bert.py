import copy
import math
import tqdm
import numpy
import torch
import torch.nn as nn
import torch.utils.data as Data
from random import *
import sys
import os

import logger
import tempfile
import tarfile
from weights.config import PreModelConfig
import shutil
import json

class BertConfig():
    def __init__(self):
        #   self.vocab_size_or_config_json_file = 30522           # 字典大小
          self.attention_probs_dropout_prob= 0.1           # 注意力处dropout值
          self.hidden_act= "gelu"                          # 隐藏层使用的激活函数
          self.hidden_dropout_prob= 0.1                    # 隐藏层处dropout的值
          self.hidden_size= 768                           # 隐藏层大小，字向量长度
          self.initializer_range= 0.02                     # bert模型初始化方差值
          self.intermediate_size= 3072                     # 前向传播隐藏层大小
          self.max_position_embeddings= 512                # 位置信息长度 512
          self.num_attention_heads= 12                     # 注意力头的个数
          self.num_hidden_layers= 12                       # encoder 层数
          self.type_vocab_size= 2                          # 句子类型，标记第一句话和第二句话
          self.vocab_size= 30522                           # 字典大小21128
          self.seq_length = 40                             # tokens总长度

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):                                  
        u = x.mean(-1, keepdim=True)                          # 最后一个维度上求均值，可以理解在字向量上
        s = (x - u).pow(2).mean(-1, keepdim=True)             # 最后一个维度上求均值，可以理解在字向量上
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):  # input_ids:(batch, seq_length)    token_type_ids:(batch, seq_length) 
        seq_length = input_ids.size(1)                                      
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)   # (seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)                        # (batch, seq_length)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        words_embeddings = self.word_embeddings(input_ids)                                   # (batch, seq_length, hidden_size)
        position_embeddings = self.position_embeddings(position_ids)                         # (batch, seq_length, hidden_size)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)                   # (batch, seq_length, hidden_size)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)                                              # 最后一个维度求归一化
        embeddings = self.dropout(embeddings)                                                    
        return embeddings      


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        self.num_attention_heads = config.num_attention_heads                            # num_attention_heads个注意力
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)  # 每一个注意力大小1024/16=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size         # all_head_size：16*64=1024
        self.query = nn.Linear(config.hidden_size, self.all_head_size)                   # (hidden_size，attention_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)                     # (hidden_size，attention_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)                   # (hidden_size，attention_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):              # q, k, v 改变形状
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) 
                                                                                   #  new_x_shape:元组(batch，seq_len, num_attention_heads,attention_head_size)
        x = x.view(*new_x_shape)                                                   # (batch，seq_len, num_attention_heads, attention_head_size)
        return x.permute(0, 2, 1, 3)                                               # (batch, num_attention_heads, seq_len, attention_head_size)

    def forward(self, hidden_states, attention_mask):                              # (batch, seq_length, hidden_size)，(batch,1,1,sqe_len)
        mixed_query_layer = self.query(hidden_states)                              # (batch, seq_len, hidden_size)
        mixed_key_layer = self.key(hidden_states)                                  # (batch, seq_len, hidden_size)
        mixed_value_layer = self.value(hidden_states)                              # (batch, seq_len, hidden_size)

        query_layer = self.transpose_for_scores(mixed_query_layer)                 # (batch, num_attention_heads, seq_len, attention_head_size)
        key_layer = self.transpose_for_scores(mixed_key_layer)                     # (batch, num_attention_heads, seq_len, attention_head_size)
        value_layer = self.transpose_for_scores(mixed_value_layer)    

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # (batch, num_attention_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # (batch, num_attention_heads, seq_len, seq_len)
        attention_scores = attention_scores + attention_mask                       # (batch, num_attention_heads, seq_len, seq_len)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)                     # (batch, num_attention_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)                            # (batch, num_attention_heads, seq_len, seq_len)

        context_layer = torch.matmul(attention_probs, value_layer)                 # (batch, num_attention_heads, seq_len, attention_head_size)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()             # (batch, seq_len，num_attention_heads, attention_head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)# (batch, seq_len，hidden_size)
        context_layer = context_layer.view(*new_context_layer_shape)               # (batch, seq_len，hidden_size)
        return context_layer                                                       # (batch, seq_len，hidden_size)

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):                   # (batch, seq_len，hidden_size),(batch, seq_len，hidden_size)
        hidden_states = self.dense(hidden_states)                     # (batch, seq_len，hidden_size)
        hidden_states = self.dropout(hidden_states)                   # (batch, seq_len，hidden_size)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # (batch, seq_len，hidden_size)
        return hidden_states                                          # (batch, seq_len，hidden_size)


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):  # hidden_states:(batch, seq_length, hidden_size)，attention_mask:(batch,1,1,sqe_len)
        self_output = self.self(input_tensor, attention_mask)         # (batch, seq_len，hidden_size)
        attention_output = self.output(self_output, input_tensor)     # (batch, seq_len，hidden_size)
        return attention_output                                       # (batch, seq_len，hidden_size)


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):                               # (batch, seq_len，hidden_size)
        hidden_states = self.dense(hidden_states)                   # (batch, seq_len，intermediate_size)
        hidden_states = self.intermediate_act_fn(hidden_states)     # (batch, seq_len，intermediate_size)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):                 # (batch, seq_len，intermediate_size)
        hidden_states = self.dense(hidden_states)                   # (batch, seq_len，hidden_size)
        hidden_states = self.dropout(hidden_states)                 # (batch, seq_len，hidden_size)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)# (batch, seq_len，hidden_size)
        return hidden_states                                        # (batch, seq_len，hidden_size)

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):  # hidden_states:(batch, seq_length, hidden_size)，attention_mask:(batch,1,1,sqe_len)
        attention_output = self.attention(hidden_states, attention_mask)      # (batch, seq_len，hidden_size)
        intermediate_output = self.intermediate(attention_output)             # (batch, seq_len，intermediate_size)
        layer_output = self.output(intermediate_output, attention_output)     # (batch, seq_len，hidden_size)
        return layer_output                                                   # (batch, seq_len，hidden_size)

    
class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):  # hidden_states:(batch, seq_length, hidden_size)，attention_mask:(batch,1,1,sqe_len)      
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)        # (batch, seq_len，hidden_size)
            if output_all_encoded_layers:                                      # 输出每一层的内容
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:                                      # 输出最后一层的内容                  
            all_encoder_layers.append(hidden_states)                           # (batch, seq_len，hidden_size)
        return all_encoder_layers  


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):                          # (batch, seq_length, hidden_size)
        first_token_tensor = hidden_states[:, 0]               # 每个句子的第一个token (batch, hidden_size)
        pooled_output = self.dense(first_token_tensor)         # (batch, hidden_size)
        pooled_output = self.activation(pooled_output)         # (batch, hidden_size)
        return pooled_output                                   # (batch, hidden_size)


class BertPreTrainedModel(nn.Module):
    #处理权重和下载、加载模型
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        # 初始化权重
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
       # 预先训练的模型下载并缓存预先训练的模型文件。
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PreModelConfig.PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PreModelConfig.PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        try:
            resolved_archive_file = PreModelConfig.cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PreModelConfig.PRETRAINED_MODEL_ARCHIVE_MAP.keys()),archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        config_file = os.path.join(serialization_dir, PreModelConfig.CONFIG_NAME)   # 加载 config文件
        if not os.path.exists(config_file):
            config_file = os.path.join(serialization_dir, PreModelConfig.BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, PreModelConfig.WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # weights_path = os.path.join(serialization_dir, PreModelConfig.TF_WEIGHTS_NAME)
            # return load_tf_weights_in_bert(model, weights_path)
            logger.error("Loading a model from a TensorFlow checkpoint is not supported yet.")
            return None
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
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
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)                        # (batch,1,1,sqe_len)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # (batch,1,1,sqe_len)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0                      # 只计算被标记的位置(batch,1,1,sqe_len)
        embedding_output = self.embeddings(input_ids, token_type_ids)                             # (batch, seq_length, hidden_size)
        encoded_layers = self.encoder(embedding_output,                                           # (batch, seq_length, hidden_size)
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)
        sequence_output = encoded_layers[-1]                                                      # 取出encoder最后一层输出(batch, seq_length, hidden_size)           
        pooled_output = self.pooler(sequence_output)                                              # 返回CLS的特征(batch, hidden_size)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output            #  返回最后一层的数据：(batch, seq_length, hidden_size),返回CLS的特征(batch, hidden_size)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):                                     # bert输出(batch, seq_length, hidden_size)
        hidden_states = self.dense(hidden_states)                         # bert输出(batch, seq_length, hidden_size)
        hidden_states = self.transform_act_fn(hidden_states)              # 激活函数
        hidden_states = self.LayerNorm(hidden_states)                     # 归一化
        return hidden_states                                              # (batch, seq_length, hidden_size)

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):                                      # bert输出(batch, seq_length, hidden_size)
        hidden_states = self.transform(hidden_states)                      # (batch, seq_length, hidden_size)
        hidden_states = self.decoder(hidden_states) + self.bias            # (batch, seq_length,vocab_size)   
        return hidden_states                                               # (batch, seq_length,vocab_size) 

class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):                # bert输出(batch, seq_length, hidden_size),CLS的特征(batch, hidden_size)
        prediction_scores = self.predictions(sequence_output)         # (batch, seq_length,vocab_size) 
        seq_relationship_score = self.seq_relationship(pooled_output) # (batch, 2)
        return prediction_scores, seq_relationship_score              # (batch, seq_length,vocab_size) ,(batch, 2)

    
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,output_all_encoded_layers=False)
        # 返回encoder最后一层(batch, seq_length, hidden_size),返回CLS的特征(batch, hidden_size)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)    # (batch, seq_length, vocab_size) ,(batch, 2)
        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)                                     # 忽略标签为-1的loss
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
                                                                                                # 计算Mask的loss的平均值                
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
                                                                                                # 计算两句话是否连续
            total_loss = masked_lm_loss + next_sentence_loss                                    # 俩个loss加起来
            return total_loss
        else:
            return prediction_scores, seq_relationship_score  