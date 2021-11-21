 

# 写在前面

## 使用方法：

1. 该脚本为测试算法脚本，没有训练

2. 邀请+信息确认过程有多个步骤，多种模式，需要多次调用该脚本。完整流程以及对应的标志如下：

   邀请  ： INVITE

   信息确认问题1 模式1 :  INFO_1_1

   信息确认问题1 模式2 :  INFO_1_2

   信息确认问题2 ： INFO_2_1

   信息确认问题3 模式1：INFO_3_1

   信息确认问题3 模式2：INFO_3_2

## Multi-classifier Naive Bayers

朴素贝叶斯短文本分类算法，输入包含文本内容和分类的文件，调用jieba分词，输出分类结果和正确率。

#### 1. 安装包：

1.1 安装pandas: 参考https://www.pypandas.cn/docs/installation.html#%E9%80%9A%E8%BF%87miniconda%E5%AE%89%E8%A3%85

```python
 pip install pandas
 pip install numpy
```

1.2 安装分词工具jieba：

```python
pip install jieba
```

1.3 安装算法包sklearn：

```python
pip install numpy

pip install matplotlib

pip install scipy

pip install sklearn
```



#### 2.运行文本分类测试脚本

```shell
python invite_and_info_Test.py 模式 文本内容

# 比如：
python invite_and_info_Test.py INFO_1_1 你发短信好吧
```

识别结果

```
(base) jingyingdeMacBook-Pro:info_and_invite jingyinghuang$ python invite_and_info_Test.py INVITE 我没时间
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/pd/9srmrd_d29n8yt74qw57r94h0000gn/T/jieba.cache
Loading model cost 0.603 seconds.
Prefix dict has been built successfully.
预测文本分词结果： [['我', '没', '时间']]
预测结果类别： ['0202']
预测结果可信度（0-1）： [0.92015884]
```

识别结果

```
python invite_and_info_Test.py INFO_1_1 160099艾特

Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/pd/9srmrd_d29n8yt74qw57r94h0000gn/T/jieba.cache
Loading model cost 0.572 seconds.
Prefix dict has been built successfully.
预测文本分词结果： [['160099', '艾', '特']]
预测结果类别： ['110101']
预测结果可信度（0-1）： [0.61455188]

python invite_and_info_Test.py INFO_1_1 发微信吧

Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/pd/9srmrd_d29n8yt74qw57r94h0000gn/T/jieba.cache
Loading model cost 0.571 seconds.
Prefix dict has been built successfully.
预测文本分词结果： [['发', '微', '信', '吧']]
预测结果类别： ['120402']
预测结果可信度（0-1）： [0.9986466]
```

未识别结果

```
python invite_and_info_Test.py INFO_1_1 未识别

Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/pd/9srmrd_d29n8yt74qw57r94h0000gn/T/jieba.cache
Loading model cost 0.549 seconds.
Prefix dict has been built successfully.
预测文本分词结果： [['未', '识别']]
预测结果类别： 未识别,130101
预测结果可信度（0-1）： [0.14935065]
```



#### 3 注意事项

！无法处理无意义的句子：如果输入无意义的句子，也就是不包含在训练数据的词或句子，算法并不一定能识别为“未识别”



