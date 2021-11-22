# -*- coding: UTF-8 -*-
import os
import random
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import sys
import numpy as np
import jieba
import pickle 
    
    
def textParser(text):
    """
    :param text:
    :return:
    """
    words = text.split("/ ")
    words = [word for word in words if word !=" "]
    return words

def loadSMSData(fileName):
    """
    :param fileName:
    :return:
    """
    f = open(fileName)
    classCategory = []
    smsWords = []
    for line in f.readlines():
        linedatas = line.strip().split('\t')
        classCategory.append(linedatas[0])
        words = textParser(linedatas[1])
        smsWords.append(words)
    return smsWords, classCategory

"""
函数说明:中文文本处理
Parameters:
    folder_path - 文本存放的路径
    test_size - 测试集占比，默认占所有数据集的百分之20
Returns:
    all_words_list - 按词频降序排序的训练集列表
    train_data_list - 训练集列表
    test_data_list - 测试集列表
    train_class_list - 训练集标签列表
    test_class_list - 测试集标签列表
"""
def TextProcessing_2(folder_path, test_size=0.2):
            
    data_list,class_list = loadSMSData(folder_path)
    
    data_class_list = list(zip(data_list, class_list)) 
    random.shuffle(data_class_list)  
    index = int(len(data_class_list) * test_size) + 1  

    train_list = data_class_list[index:]  # 训练集
    test_list = data_class_list[:index]  # 测试集
    # 1.从不同类别中取相同数目的测试集

    train_data_list, train_class_list = zip(*train_list)  # 训练集解压缩
    test_data_list, test_class_list = zip(*test_list)  # 测试集解压缩

    all_words_dict = {}  # 统计训练集词频
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    # 根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key=lambda f: f[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)  # 解压缩
    all_words_list = list(all_words_list)  # 转换成列表
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

"""
函数说明:读取文件里的内容，并去重
Parameters:
    words_file - 文件路径
Returns:
    words_set - 读取的内容的set集合
"""
def MakeWordsSet(words_file):
    words_set = set()  # 创建set集合
    with open(words_file, 'r', encoding='utf-8') as f:  # 打开文件
        for line in f.readlines():  # 一行一行读取
            word = line.strip()  # 去回车
            if len(word) > 0:  # 有文本，则添加到words_set中
                words_set.add(word)
    return words_set  # 返回处理结果


"""
函数说明:文本特征选取
Parameters:
    all_words_list - 训练集所有文本列表
    deleteN - 删除词频最高的deleteN个词
    stopwords_set - 指定的结束语
Returns:
    feature_words - 特征集
"""
def words_dict(all_words_list, deleteN, stopwords_set=set()):
    feature_words = []  # 特征列表
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:  
            break
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words


"""
函数说明:根据feature_words将文本向量化
Parameters:
    train_data_list - 训练集
    test_data_list - 测试集
    feature_words - 特征集
Returns:
    train_feature_list - 训练集向量化列表
    test_feature_list - 测试集向量化列表
"""
def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list 


"""
Parameters:
    train_feature_list - 训练集向量化的特征文本
    test_feature_list - 测试集向量化的特征文本
    train_class_list - 训练集分类标签
    test_class_list - 测试集分类标签
Returns:
    test_accuracy - 分类器精度
"""
def TextClassifier(filename,train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list, train_class_list) 
    # save object
    with open(filename, 'wb') as filehandler:
        pickle.dump(classifier, filehandler)
    # get class
    test_predict_class = classifier.predict(test_feature_list)
    # get prob
    test_predict_prob = classifier.predict_proba(test_feature_list)

    test_accuracy = classifier.score(test_feature_list, test_class_list)
    return test_accuracy,test_predict_class,test_predict_prob


"""
 write the result of classifer to file
"""
def WriteTestResult(test_data_list,test_class_list,test_predict_class,test_predict_prob):
    col1="test_data"
    col2="test_class"
    col3="predict_class"
    col4="predict_prob"
    df=pd.DataFrame(columns=[col1,col2,col3])
    df[col1]=test_data_list
    df[col2]=test_class_list
    df[col3]=test_predict_class
    # print(test_predict_prob[:, 1])
    test_predict_prob=np.sort(test_predict_prob,axis=1)
    df[col4]=test_predict_prob[:, -1]#-test_predict_prob[:, 2]
    return df


# def TestOneSample(train_feature_list, train_class_list,sepstring):
#     # read model
#     classifier = MultinomialNB().fit(train_feature_list, train_class_list) 
#     # class
#     test_predict_class = classifier.predict(sepstring)
#     # prob
#     test_predict_prob = classifier.predict_proba(sepstring)
#     return test_predict_class,test_predict_prob

"""
write all words to a file, used for test text feature
"""
def WriteWordList(wordlistfile,all_words_list):
    # save all_word_list
    with open(wordlistfile,'w') as writer:
        for i in range(len(all_words_list)):
            writer.write(all_words_list[i]+"\n")

if __name__ == '__main__':
    level="A24"
    for level in ["A","A1","A2","A21","A22","A23","A24"]:
        filename="traindata_"+level+"_Tokenization_fullmode"
        # filename=sys.argv[1]
        file=filename+".csv"
        folder_path2 = "./"+file
       
        all_words_list2, train_data_list2, test_data_list2, train_class_list2, test_class_list2 = TextProcessing_2(folder_path2,test_size=0.1)
        # print(test_data_list2[:2])
        test_accuracy_list = []
    
        # feature_words2 = words_dict(all_words_list2, 0,)
        feature_words2=all_words_list2
        wordlistfile=level+"_All_word_list.txt"
        WriteWordList(wordlistfile,feature_words2)
    
        train_feature_list2, test_feature_list2 = TextFeatures(train_data_list2, test_data_list2, feature_words2)
    
        modelfilename=level+"_saved_model_file"
        test_accuracy2,test_predict_class2,test_predict_prob2 = TextClassifier(modelfilename,train_feature_list2, test_feature_list2, train_class_list2, test_class_list2)
        # print(test_feature_list2[:2])
    
        df = WriteTestResult(test_data_list2,test_class_list2,test_predict_class2,test_predict_prob2)
        df.to_csv(filename+".predict.prob.csv",sep=",",encoding='utf_8_sig')
        
        test_accuracy_list.append(test_accuracy2)
        ave = lambda c: sum(c) / len(c)
        print(ave(test_accuracy_list))

