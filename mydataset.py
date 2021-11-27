import torch
import torch.utils.data as Data
import numpy
from tqdm import tqdm
import ipdb
import config

def load_dataset(path, pad_size=32):
    contents = []
    word2idx = {}
    idx2word = {}
    with open("./weights/vocab.txt", 'r', encoding='UTF-8') as f:
        idx2word = {idx: line.strip() for idx, line in  enumerate(tqdm(f))}
        word2idx = {idx2word[key]: key for key in  idx2word}
    with open(path, 'r', encoding='UTF-8') as f:
        for line in tqdm(f):
            token_ids = []
            # ipdb.set_trace()
            lin = line.strip()
            if not lin:
                continue
            content, label = lin.split('\t')
            token_ids.append (word2idx['[CLS]'])
            for key in content:
                token_ids.append(word2idx.get(key, 0))
            seq_len = len(token_ids)
            mask = []
            if pad_size:
                if seq_len < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - seq_len)
                    token_ids += ([0] * (pad_size - seq_len))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append((numpy.array(token_ids), int(label), seq_len, numpy.array(mask)))
    return contents


train = load_dataset("./data/train.txt")
dev = load_dataset("./data/dev.txt")
test = load_dataset("./data/test.txt")

train_input_ids, train_label, seq_len, train_attention_mask = zip(*train)
dev_input_ids, dev_label, seq_len, dev_attention_mask = zip(*dev)
test_input_ids, test_label, seq_len, test_attention_mask = zip(*test)

train_input_ids, train_label, train_attention_mask = torch.tensor(train_input_ids), torch.tensor(train_label), torch.tensor(train_attention_mask)
dev_input_ids, dev_label, dev_attention_mask = torch.tensor(dev_input_ids), torch.tensor(dev_label), torch.tensor(dev_attention_mask)
test_input_ids, test_label, test_attention_mask = torch.tensor(test_input_ids), torch.tensor(test_label), torch.tensor(test_attention_mask)

class MyDataSet(Data.Dataset):
  def __init__(self, input_ids, label, attention_mask):
    self.input_ids = input_ids
    self.label = label
    self.attention_mask = attention_mask

  def __len__(self):
    return len(self.input_ids)
      
  def __getitem__(self, idx):
    return self.input_ids[idx], self.label[idx], self.attention_mask[idx]

train_loader = Data.DataLoader(MyDataSet(train_input_ids, train_label, train_attention_mask), config.batch_size, True)
dev_loader = Data.DataLoader(MyDataSet(dev_input_ids, dev_label, dev_attention_mask), config.batch_size, True)
test_loader = Data.DataLoader(MyDataSet(test_input_ids, test_label, test_attention_mask), config.batch_size, True)