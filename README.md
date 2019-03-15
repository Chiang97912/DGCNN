# DGCNN
DGCNN，全名为Dilate Gated Convolutional Neural Network，即“膨胀门卷积神经网络”，顾名思义，融合了两个比较新的卷积用法：膨胀卷积、门卷积，并增加了一些人工特征和trick，最终使得模型在轻、快的基础上达到最佳的效果。

该仓库是使用Tensorflow对苏剑林的博客：[基于CNN的阅读理解式问答模型：DGCNN](https://kexue.fm/archives/5409)中提出的DGCNN模型的实现。具体的网络结构各位同学可以前往苏神博客一探究竟。



# 数据集

因为DGCNN属于机器阅读理解式问答系统，所以本模型可以使用SQuAD数据集（英文），同样也可以使用SOGOU问答比赛提供的数据集，但是这个项目我使用的是我自己根据WebQA处理的数据集（我们实验室的NLP小组共同使用），如果有感兴趣的同学可以给我发e-mail。



# 词向量

词向量是使用gensim根据中文维基百科语料库训练的60维词向量，网上也有很多资源，同样有感兴趣的同学可以给我发e-mail。



# 安装

```
git clone https://github.com/Chiang97912/DGCNN.git
```



# 使用

## 怎么训练DGCNN?

```
python train.py
```

## 怎么测试DGCNN?

```
python test.py
```

