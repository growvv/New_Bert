import torch
import torch.nn as nn
import torch.utils.data as Data
from tqdm.std import tqdm

from mydataset import train_loader, dev_loader, test_loader
from bert import BertConfig, BertForPreTraining
from draw_loss import draw_loss
import ipdb
import config

# bert_config = BertConfig.from_json_file("./weights/config.json")
bert_config = BertConfig()
# ipdb.set_trace()
print("Building PyTorch model from configuration: {}".format(str(bert_config)))
model = BertForPreTraining(bert_config).to(config.device)
learnrate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learnrate)
loss_fct = nn.CrossEntropyLoss() 
#optimizer = torch.optim.SGD(model.parameters(), lr=learnrate, momentum=0.99)
model.load_state_dict(torch.load('./weights/pytorch_model.bin'), strict=False)  # use not strict, because (weight,bias) is corresponding to (gamma,beta) in pre-train model 
# print(model)

# 第一部分：训练
train_loss = []
train_accuracy =[]
data_num = len(train_loader) * config.batch_size
for epoch in range(config.epochs):
    all_loss = 0
    accuracy =0
    for input_ids, label, attention_mask in tqdm(train_loader):
        input_ids = input_ids.to(config.device)
        label = label.to(config.device)
        attention_mask = attention_mask.to(config.device)
        _, score_label = model(input_ids, attention_mask=attention_mask, next_sentence_label=label)
        accuracy += torch.sum(score_label.argmax(dim = 1).view(-1) == label.view(-1)).item()
        loss = loss_fct(score_label, label)
        # print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
        all_loss += loss.item()
        train_loss.append(loss.item())
        train_accuracy.append(accuracy)
    print('Epoch Train:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(all_loss/(len(train_loader)*config.batch_size)),'acc =', '{:.6f}'.format(accuracy/(len(train_loader)*config.batch_size)))
    draw_loss(train_loss, str(epoch) +'train_loss.png', line=True)
    draw_loss(train_accuracy, str(epoch) +'train_accuracy.png', line=True)

    # 第二部分：验证
    all_loss = 0
    accuracy = 0
    for input_ids, label, attention_mask in tqdm(dev_loader):
        with torch.no_grad():
            input_ids = input_ids.to(config.device)
            label = label.to(config.device)
            attention_mask = attention_mask.to(config.device)
            _, score_label = model(input_ids, attention_mask=attention_mask, next_sentence_label=label)
            accuracy += torch.sum(score_label.argmax(dim = 1).view(-1) == label.view(-1)).item()
            loss = loss_fct(score_label, label)
            all_loss += loss.item()
    print('Epoch  dev :', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(all_loss/(len(dev_loader)*config.batch_size)),'acc =', '{:.6f}'.format(accuracy/(len(dev_loader)*config.batch_size)))

# 第三部分：测试
accuracy = 0
all_loss = 0
for input_ids, label, attention_mask in tqdm(test_loader):
    with torch.no_grad():
        input_ids = input_ids.to(config.device)
        label = label.to(config.device)
        attention_mask = attention_mask.to(config.device)
        _, score_label = model(input_ids, attention_mask=attention_mask, next_sentence_label=label)
        accuracy += torch.sum(score_label.argmax(dim = 1).view(-1) == label.view(-1)).item()
        loss = loss_fct(score_label, label)
        all_loss += loss.item()
print('loss =', '{:.6f}'.format(all_loss/(len(test_loader)*config.batch_size)),'  acc =', '{:.6f}'.format(accuracy/(len(test_loader)*config.batch_size)))