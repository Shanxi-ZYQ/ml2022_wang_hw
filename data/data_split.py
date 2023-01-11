import os
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#划分数据集(采用7:2:1)
'''
label_data:数据集的标签
text_data:数据集的文本
shuffle:在划分前是否对数据洗牌
random_state:随机种子,如果更改则每次划分结果不同

'''
def data_split(label_data,text_data,validation_size=0.1,test_size=0.2,shuffle=False,random_state = 2022):
    #第一次划分出测试集
    label_,label_test,text_,text_test = train_test_split(label_data,text_data,test_size=test_size,shuffle=shuffle,random_state=2022)
    #在从剩余的数据中划分出验证集
    vaild_size = validation_size / (1.0-test_size)
    #划分出验证集
    label_train, label_vaild, text_train, text_vaild = train_test_split(label_,text_,test_size=vaild_size,shuffle=shuffle,random_state=2022)
    return label_train, label_vaild, label_test, text_train, text_vaild, text_test

if __name__ == '__main__':
    #数据集路径
    path = "train/"
    pd_all = pd.read_csv(os.path.join(path,"train.csv"), sep = '\t',error_bad_lines=False)
    pd_all = shuffle(pd_all)
    label_data,text_data = pd_all.label,pd_all.text

    label_train, label_vaild, label_test, text_train, text_vaild, text_test = data_split(label_data,text_data,0.1,0.2)

    train = pd.DataFrame({'label':label_train, 'text':text_train})
    train.to_csv("train/train_split.csv", index=False, encoding='utf-8',sep='\t')
    
    vaild = pd.DataFrame({'label':label_vaild, 'text':text_vaild})
    vaild.to_csv("train/vaild_split.csv", index=False, encoding='utf-8',sep='\t')
    
    test = pd.DataFrame({'label':label_test, 'text':text_test})
    test.to_csv("train/test_split.csv", index=False, encoding='utf-8',sep='\t')