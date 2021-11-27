## Bert迁移学习实现评语情感分类

### 自定义Bert

虽然可以从huggingface、pytorch_pretrained_bert等框架中导入Bert，但是我们也可以自己实现Bert.

参考 [pytorch_pretrained_bert](https://github.com/SVAIGBA/TwASP/blob/c757769f377de4edc48623cff8cb446f936d14d0/pytorch_pretrained_bert/modeling.py) 中的Bert实现。

Other文件夹下有Bert的单独测试，可以参考。

### 评语情感分类

数据集、数据预处理、模型训练、模型评估都参考自:

[【BERT：一切过往， 皆为序章】BERT迁移学习 | 情感分类 | 微调 | Pytorch代码讲解 | 代码可运行](https://zhuanlan.zhihu.com/p/405550024)

#### 训练输出

#### 评估输出
为啥我的我直接下的bert accuracy 才 0.6
作者的有0.9，不是用的普通的权重??
