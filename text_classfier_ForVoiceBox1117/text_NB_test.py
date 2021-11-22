# -*- coding: UTF-8 -*-
import os
import random
from sklearn.naive_bayes import MultinomialNB
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

"""
单个句子测试
"""
def TestOneSample(filename,sepstring):
    # read model
    # classifier = MultinomialNB().fit(train_feature_list, train_class_list) 
    with open(filename, 'rb') as filehandler:
        classifier = pickle.load(filehandler)
    # class
    test_predict_class = classifier.predict(sepstring)
    # prob
    test_predict_prob = classifier.predict_proba(sepstring)
    test_predict_prob=np.sort(test_predict_prob,axis=1)
    return test_predict_class,test_predict_prob[:, -1]

"""
从文本读取所有词
"""
def ReadWordList(wordlistfile):
    all_word_list=[]
    with open(wordlistfile,'r') as reader:
        for line in reader:
            all_word_list.append(line.rstrip())
    return all_word_list

"""
文本向量化
"""
def TextFeaturesTest(test_data_list, feature_words):
    def text_features(text, feature_words):
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features

    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return test_feature_list 

"""
预测主函数
"""
def predict(level):
    all_words_list2=ReadWordList(level+"_All_word_list.txt")
    # feature_words2 = words_dict(all_words_list2, 0,)
    feature_words2=all_words_list2
    # print(feature_words2[:2])

    #--single test
    # inputstring="你发短信好吧"
    inputstring=sys.argv[1]
    seg = jieba.cut(inputstring, cut_all=True)
    sepstring="/ ".join(seg)
    sepstring=textParser(sepstring)
    sepstring_list=[sepstring]
#    print("预测文本分词结果：",sepstring_list)
    test_one_feature = TextFeaturesTest(sepstring_list, feature_words2)
    # print(test_one_feature)
    modelfilename=level+"_saved_model_file"
    predict_class,predict_prob=TestOneSample(modelfilename,test_one_feature)
    if predict_prob<0.3:
        predict_class=["未识别"]
    
    return predict_class,predict_prob



if __name__ == '__main__':

    predict_class_list=[]
    predict_class,predict_prob = predict("A")
    predict_class_list.append(predict_class[0])
    
    
    if predict_class=="A1":
        predict_class,predict_prob = predict("A1")
        predict_class_list.append(predict_class[0])
        
    elif predict_class=="A2":
        predict_class,predict_prob = predict("A2")
        predict_class_list.append(predict_class[0])
        if predict_class!="未识别":
            predict_class,predict_prob = predict(predict_class[0])
            predict_class_list.append(predict_class[0])
        
    flag=True
    for i in predict_class_list[::-1]:
        if i!="未识别":
            predict_class=i
            print("预测结果类别：",predict_class)
#            print("预测结果类别：",predict_class_list)
#            print("预测结果可信度（0-1）：",predict_prob)
            flag=False
            break
    if flag:
        print("预测结果类别：","未识别")
#        print("预测结果类别：",predict_class_list)