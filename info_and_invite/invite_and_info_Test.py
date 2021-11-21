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
测试模型
"""
def runTest(sepstring_list,wordListfile,modelfilename):
    all_words_list2=ReadWordList(wordListfile)
    # feature_words2 = words_dict(all_words_list2, 0,)
    feature_words2=all_words_list2
    # print(feature_words2[:2]
    test_one_feature = TextFeaturesTest(sepstring_list, feature_words2)
    # print(test_one_feature)
    predict_class,predict_prob=TestOneSample(modelfilename,test_one_feature)
    return predict_class,predict_prob

"""
输入句子分词
"""
def SepText(inputstring):
    seg = jieba.cut(inputstring, cut_all=True)
    sepstring="/ ".join(seg)
    sepstring=textParser(sepstring)
    sepstring_list=[sepstring]
    print("预测文本分词结果：",sepstring_list)
    return sepstring_list
"""
邀请测试
"""
def invite_test(inputstring):
    sepstring_list = SepText(inputstring)
    path="invite/"
    #--一次分类
    wordListfile=path+"All_word_list_level2.txt"
    modelfilename=path+"saved_model_file_level2"
    predict_class,predict_prob=runTest(sepstring_list,wordListfile,modelfilename)

    #--二次分类
    if (predict_class[0]=="0303")|(predict_class[0]=="0304"):
        wordListfile=path+"All_word_list_level3.txt"
        modelfilename=path+"saved_model_file_level3"
        predict_class,predict_prob=runTest(sepstring_list,wordListfile,modelfilename)
        
    if predict_prob<0.15:
        predict_class="未识别,"+predict_class[0]
    return predict_class,predict_prob

"""
信息确认测试
"""
def info_test(inputstring,inputType):
    #--single test
    # inputstring="你发短信好吧"
    typeList=["INFO_1_1","INFO_1_2","INFO_2_1","INFO_3_1","INFO_3_2"]
    if inputType in typeList:
        step=inputType.split("_")[1]
        mode=inputType.split("_")[2]
    else:
        print("Input is wrong.")
        return

    sepstring_list = SepText(inputstring)
    path="info/"
    all_words_list2=ReadWordList(path+"All_word_list_"+step+"_"+mode+".txt")
    # feature_words2 = words_dict(all_words_list2, 0,)
    feature_words2=all_words_list2
    # print(feature_words2[:2])
    test_one_feature = TextFeaturesTest(sepstring_list, feature_words2)
    # print(test_one_feature)
    modelfilename=path+"saved_model_file_"+step+"_"+mode
    predict_class,predict_prob=TestOneSample(modelfilename,test_one_feature)
    if predict_prob<0.15:
        predict_class="未识别,"+predict_class[0]
    return predict_class,predict_prob


if __name__ == '__main__':
    inputType=sys.argv[1]
    inputstring=sys.argv[2]
    if inputType=="INVITE":
        predict_class,predict_prob = invite_test(inputstring) 
    elif inputType.startswith("INFO"):
        predict_class,predict_prob = info_test(inputstring,inputType) 
    else:
        print("Input is wrong.")

    print("预测结果类别：",predict_class)
    print("预测结果可信度（0-1）：",predict_prob)

