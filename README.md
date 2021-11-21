# Multistage_Text_Classification
Develop a Multistage Text Classifier for multiple rounds of dialogue.

## 1. Program objective:

Develop an intelligent robot that can automatically call customers, invite them to attend conferences, and answer their questions.

- Let the machine understand customers' questions and provide appropriate answers: the semantic analysis task was transformed into a classification task, where a Multi-stages Text Classifier Model determined the category of customer's answer based on the text information, and the system set the corresponding discourse for all answer categories, thus enabling multiple rounds of dialogue.

- Solve Multi-label text classification problem: adopted Naive Bayesian (NB) algorithm to build a Multi-stages Text Classifier Model for short sentences classification, based on semantic similarity within the given category.


## 2. Usage.

1. This script is an algorithm run/test script and does not contain a training script

2. The invitation + message confirmation process has multiple steps and multiple modes, requiring multiple calls to this script. The complete process and the corresponding flags are as follows.

   INVITE : INVITE

   Message Confirmation Question 1 Mode 1 : INFO_1_1

   Message Confirmation Question 1 Mode 2 : INFO_1_2

   Message confirmation question 2 : INFO_2_1

   Information confirmation question 3 Mode 1 : INFO_3_1

   Information confirmation question 3 Mode 2 : INFO_3_2
   
#### 2.1. Install package：

1.1 Install pandas: ref: https://www.pypandas.cn/docs/installation.html#%E9%80%9A%E8%BF%87miniconda%E5%AE%89%E8%A3%85

```python
 pip install pandas
 pip install numpy
```

1.2 Install Tokenization tool for Chinese: jieba

```python
pip install jieba
```

1.3 Install sklearn：

```python
pip install numpy

pip install matplotlib

pip install scipy

pip install sklearn
```
#### 2.2. Run the text classification test script

```shell
python invite_and_info_Test.py 模式 文本内容

# 比如：
python invite_and_info_Test.py INFO_1_1 你发短信好吧
```

Result Demo: Recognized

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
Result Demo: Recognized

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

Result Demo: Unrecognized

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

## 3 Caution

! Cannot handle nonsensical sentences: if a nonsensical sentence is entered, i.e. a word or sentence that is not included in the training data, the algorithm does not necessarily recognize it as "not recognized"
