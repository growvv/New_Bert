import pandas
from random import *
import re
import ipdb

# ipdb.set_trace()
f = pandas.read_csv("./data/Data.csv")
x = pandas.concat((f.iloc[:, 1:], f.iloc[:, :1]),axis = 1, ignore_index = False)
x = list(x.values)
shuffle(x)
y = []
g = []
for i in x:
    y = re.sub("[.,!?\\-。，？~！\" ]", "", str(i[0]))
    y = y + "\t" + str(i[1])
    g.append(y)
g = pandas.DataFrame(g)
# print(len(g))  # 11987
train = g[:8000]
dev = g[8000:10000]
test = g[10000:]
train.to_csv("./data/train.txt", header=0, index=0)
dev.to_csv("./data/dev.txt", header=0, index=0)
test.to_csv("./data/test.txt", header=0, index=0)