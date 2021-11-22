 

# 写在前面

## 使用方法：

1. 


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
python text_NB_test.py 文本内容

# 比如：
python text_NB_test.py 你们公司上周在深圳有哪些讲座

```

识别结果

```
(base) jingyingdeMacBook-Pro:text_classfier_ForVoiceBox1117 jingyinghuang$ python text_NB_test.py 你们公司上周在深圳有哪些讲座
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/pd/9srmrd_d29n8yt74qw57r94h0000gn/T/jieba.cache
Loading model cost 0.567 seconds.
Prefix dict has been built successfully.
预测结果类别： A2402
```

识别结果

```
(base) jingyingdeMacBook-Pro:text_classfier_ForVoiceBox1117 jingyinghuang$ python text_NB_test.py 你有哪些功能
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/pd/9srmrd_d29n8yt74qw57r94h0000gn/T/jieba.cache
Loading model cost 0.546 seconds.
Prefix dict has been built successfully.
预测结果类别： A12

(base) jingyingdeMacBook-Pro:text_classfier_ForVoiceBox1117 jingyinghuang$ python text_NB_test.py AWS干什么
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/pd/9srmrd_d29n8yt74qw57r94h0000gn/T/jieba.cache
Loading model cost 0.547 seconds.
Prefix dict has been built successfully.
预测结果类别： A2101
```

需要改进的结果

```
(base) jingyingdeMacBook-Pro:text_classfier_ForVoiceBox1117 jingyinghuang$ python text_NB_test.py AWS是什么
Building prefix dict from the default dictionary ...
Loading model from cache /var/folders/pd/9srmrd_d29n8yt74qw57r94h0000gn/T/jieba.cache
Loading model cost 0.553 seconds.
Prefix dict has been built successfully.
预测结果类别： A2401
```



#### 3 注意事项

！无法处理无意义的句子：如果输入无意义的句子，也就是不包含在训练数据的词或句子，算法并不一定能识别为“未识别”



