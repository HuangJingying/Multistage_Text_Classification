#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 14:37:37 2021

@author: jingyinghuang
"""

import jieba
import os
import pandas as pd
import matplotlib.pyplot as plt
import random

os.chdir("/Users/jingyinghuang/PycharmProjects/NLP_text_classfication/VoiceBox/text_classfier_ForVoiceBox1117")

df = pd.read_csv("./train_data.csv",dtype=str)

df.columns

df.head()
# df[['客户-应答文本']]

    
#--------jieba: split sentences to words
i=1
l=len(df)
for i in range(l):
    # 全模式
    seg_list = jieba.cut(df.loc[i,'常用表达'], cut_all=True)
    string="/ ".join(seg_list)
    #print("Full Mode: " + "/ ".join(seg_list))  
    df.loc[i,'常用表达_jieba']=string
    #for m in seg_list:
    #    print(m)
    # 精确模式
#     seg_list = jieba.cut(df.loc[i,'客户-应答文本'], cut_all=False)
#     string="/ ".join(seg_list)
# #    print("Default Mode: " + "/ ".join(seg_list))  
#     df.loc[i,'客户-应答文本-jieba_defaultmode']=string
    
    
#     seg_list = jieba.cut_for_search(df.loc[i,'客户-应答文本'])
#     string="/ ".join(seg_list)
# #    print("Default Mode: " + "/ ".join(seg_list))  
#     df.loc[i,'客户-应答文本-jieba_searchmode']=string


# print(df['客户-应答文本-jieba_fullmode'])

# df.to_csv("./语音对话-邀请场景-ai数据-jieba.csv",index=False)
 

# filename="./语音对话-邀请场景-ai数据-jieba.csv"
# df = pd.read_csv(filename,sep=",")
# print(df.columns)
"""
filter some classifies from raw data

"""
# mode="defaultmode"
# mode="searchmode"
mode="fullmode"
# traindata=df[[i for i in df.columns if ('客户-应答文本-jieba_'+mode in i)|('应答编码（二级）' in i)|('应答编码（三级）' in i)]]

#traindata.to_csv("level2_traindata.csv",sep="\t",index=False,header=False)
# tmp = traindata[~(traindata['应答编码（二级）'].isin([308]))]
# tmp.to_csv("level2_cut1_traindata.csv",sep="\t",index=False,header=False)



# traindata['应答编码（三级）'].replace("30303","30302",regex=True,inplace=True)

# #plot 303 and 304
# plt.figure()
# otherdata=traindata[(traindata['应答编码（二级）']=='303')|(traindata['应答编码（二级）']=='304')]
# count=otherdata['应答编码（三级）'].value_counts(sort=False)
# count=pd.DataFrame(count)
# count['x']=count.index
# count=count.sort_values('x')
# # count.index=count.index.astype(str)
# plt.bar(count.index, count['应答编码（三级）'])
# plt.xticks(rotation=90,fontname="Arial")#fontsize=15,
# plt.savefig("bar_number_class_level3.png")
# print(count)


# print(traindata['应答编码（二级）'])
traindata=df[[i for i in df.columns if ('第一层神经元编码' in i)|('常用表达_jieba' in i)]]

plt.figure(figsize=(8,6))
count=traindata['第一层神经元编码'].value_counts(sort=False)
count=pd.DataFrame(count)
count['x']=count.index
count=count.sort_values('x')
# count.index=count.index.astype(str)
plt.bar(count.index, count['第一层神经元编码'])
plt.xticks(rotation=90,fontname="Arial")#fontsize=15,
plt.savefig("1st_bar_number_class.png")
# print(count)

traindata.to_csv("traindata_A_Tokenization_"+mode+".csv",sep="\t",index=False,header=False)





traindata=df[[i for i in df.columns if ('第二层神经元编码' in i)|('常用表达_jieba' in i)]]

plt.figure(figsize=(8,6))
count=df['第二层神经元编码'].value_counts(sort=False)
count=pd.DataFrame(count)
count['x']=count.index
count=count.sort_values('x')
# count.index=count.index.astype(str)
plt.bar(count.index, count['第二层神经元编码'])
plt.xticks(rotation=90,fontname="Arial")#fontsize=15,
plt.savefig("2nd_bar_number_class.png")
traindata.to_csv("traindata_level2_Tokenization_"+mode+".csv",sep="\t",index=False,header=False)

for i in ["A1","A2"]:
    savedata=traindata[traindata['第二层神经元编码'].str.contains(i)]
    savedata.to_csv("traindata_"+i+"_Tokenization_"+mode+".csv",sep="\t",index=False,header=False)




traindata=df[[i for i in df.columns if ('第三层神经元编码' in i)|('常用表达_jieba' in i)]]
traindata=traindata[~traindata['第三层神经元编码'].isnull()]

plt.figure(figsize=(8,6))
count=df['第三层神经元编码'].value_counts(sort=False)
count=pd.DataFrame(count)
count['x']=count.index
count=count.sort_values('x')
# count.index=count.index.astype(str)
plt.bar(count.index, count['第三层神经元编码'])
plt.xticks(rotation=90,fontname="Arial")#fontsize=15,
plt.savefig("3rd_bar_number_class.png")
traindata.to_csv("traindata_level3_Tokenization_"+mode+".csv",sep="\t",index=False,header=False)

for i in ["A21","A22","A23","A24"]:
    savedata=traindata[traindata['第三层神经元编码'].str.contains(i)]
    savedata.to_csv("traindata_"+i+"_Tokenization_"+mode+".csv",sep="\t",index=False,header=False)



#
#
## traindata.to_csv("step3_traindata_"+mode+"_0430.csv",sep="\t",index=False,header=False)
#
## 2.删除过多（>200)的类别
#deletClass=count[count['意图编码']>150].index
#for cla in deletClass:
#    print(">150",cla)
#    deletData=traindata[traindata['意图编码']==cla]#
#    # pandas sample function
#    deletData = deletData.sample(n=180,axis=0,replace=True,random_state=None)
#    # deletData=deletData[:180]
#    traindata=traindata[traindata['意图编码']!=cla]#
#    traindata=pd.concat([traindata,deletData],axis=0)
#
#
## 3.重复过少（<100)的类别
#addClass=count[count['意图编码']<100].index
#for cla in addClass:
#    print("<100",cla)
#    addData=traindata[traindata['意图编码']==cla]#
#    n=len(addData)
#    addData=pd.concat([addData]*int(120/n),axis=0,ignore_index=True) #!concat
#    # addData=pd.concat([addData,addData],axis=0)
#    # traindata=traindata[traindata['应答编码（二级）']!=cla]#
#    traindata=pd.concat([traindata,addData],axis=0,ignore_index=True)
#    # traindata.append(addData)
#
#plt.figure(figsize=(8,6))
#count=traindata['意图编码'].value_counts(sort=False)
#count=pd.DataFrame(count)
#count['x']=count.index
#count=count.sort_values('x')
## count.index=count.index.astype(str)
#plt.bar(count.index, count['意图编码'])
#plt.xticks(rotation=90,fontname="Arial")#fontsize=15,
#plt.savefig("bar_number_class_balanced.png")
#print(count)



# traindata.loc[(traindata['应答编码（二级）']=='303')|(traindata['应答编码（二级）']=='304'),'应答编码（二级）']=traindata['应答编码（三级）']

# traindata[['客户-应答文本-jieba_'+mode,'应答编码（二级）']].to_csv("traindata_"+mode+".csv",sep="\t",index=False,header=False)


