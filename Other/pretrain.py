from modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert, BertModel
import torch
import torch.utils.data as Data
import copy
from random import *
import ipdb


data = ['有生之年',
        '愿你勇敢',
        '愿你平安',
        '愿你没有苦难',
        '活的简单',
        '愿你累了倦了有人为你分担',
        '愿你每个夜晚都会有美梦作伴',
        '愿你长路漫漫得偿所愿',
        '愿这世间烦恼从此与你无关',
        '愿你遇事全部得你心欢',
        '愿你前程似锦',
        '不凡此生']

#建立字典 编号 <--> 字 的对应关系
s = set([ i for j in data for i in j])                               # 所有的字
word2idx = {'PAD' : 0, 'CLS' : 1, 'SEP' : 2, 'MASK' : 3}             # 特殊字符
for idx, word in enumerate(s):
    word2idx[word] = idx+4                                           # 字   -> 编号
idx2word = {word2idx[key]:key for key in word2idx}                   # 编号 -> 字
vocab_size = len(idx2word)

#把句子的字变成编号
sentences = []
for sentence in data:
    tmp = []
    for i, word in enumerate(sentence):
        tmp.append(word2idx[word])
    sentences.append(tmp)

seq_length = 40
# 自定义Dataset
batch_size = 8
class MyDataSet(Data.Dataset):
    def __init__(self, data):
        self.sentences = []
        for sentence in data:
            tmp = []
            for i, word in enumerate(sentence):
                tmp.append(word2idx[word])
            self.sentences.append(tmp)
        self.sentences_len = len(self.sentences)
    def __len__(self):
        return len(self.sentences)*2-2  
    def __getitem__(self, idx):
        sentences = copy.deepcopy(self.sentences)       
        input_ids = []
        token_type_ids = []
        next_sentence_label = []

        if idx%2 == 0:
            s = [word2idx['CLS']] + sentences[int(idx//2)] + [word2idx['SEP']] + sentences[int(idx//2)+1] + [word2idx['SEP']]
            input_ids = s+[0]*(seq_length-len(s))
            token_type_ids = [0]*(1+len(sentences[int(idx//2)])+1) + [1]*(len(sentences[int(idx//2+1)])+1) + [0]*(seq_length-len(s))
            next_sentence_label = [1]
        else:
            rand = int(idx//2)+1
            while rand ==  idx//2+1:
                rand = randint(0, self.sentences_len-1)
            s =[word2idx['CLS']] + sentences[int(idx//2)] + [word2idx['SEP']] + sentences[rand] + [word2idx['SEP']]
            input_ids = s+[0]*(seq_length-len(s))
            token_type_ids = [0]*(1+len(sentences[int(idx//2)])+1) + [1]*(len(sentences[rand])+1) + [0]*(seq_length-len(s))
            next_sentence_label = [0]
            
        attention_mask = []
        masked_lm_labels = []
        for pos, value in enumerate(input_ids):
            rand = random()
            if value == 0:
                attention_mask.append(0) 
            else:
                attention_mask.append(1)
            if value != 0 and value != 1 and value != 2 and rand < 0.15:   
                    masked_lm_labels.append(input_ids[pos]) 
                    if rand < 0.15*0.8:
                        input_ids[pos] = word2idx["MASK"]
                    elif rand > 0.15*0.9:
                        input_ids[pos] = randint(4, vocab_size-1)           
            else:  
                masked_lm_labels.append(-1)          
        input_ids = torch.tensor(input_ids)
        token_type_ids = torch.tensor(token_type_ids)
        next_sentence_label = torch.tensor(next_sentence_label)
        attention_mask = torch.tensor(attention_mask)
        masked_lm_labels = torch.tensor(masked_lm_labels)
        return input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label

data_loader = Data.DataLoader(MyDataSet(data), batch_size, False)


# for idx, batch in enumerate(data_loader):
#     ipdb.set_trace()
#     print(idx, batch)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = BertConfig.from_json_file("./weights/bert-base-uncased-config.json")
model = BertForPreTraining(config).to(device)
learnrate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learnrate, momentum=0.99)

epochs = 100
for epoch in range(epochs):
    for input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label in data_loader:
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        masked_lm_labels = masked_lm_labels.to(device)
        next_sentence_label = next_sentence_label.to(device)
        loss = model(input_ids, token_type_ids, attention_mask, masked_lm_labels, next_sentence_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))