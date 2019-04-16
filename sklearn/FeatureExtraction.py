# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/4/15
@description: 特征提取
"""
import numpy as np
import pandas as pd


## 无序分类变量的特征提取/ onehot 编码
## 只能处理String类型的dict数据
## 适用范围太小，不如用 sklearn.preprocessing.OneHotEncoder
## 默认用 scipy.sparse 存储 可用.toarray转换为numpy
from sklearn.feature_extraction import DictVectorizer
vec  = DictVectorizer()
instances = [{'city':'北京'},{'city':'南京'},{'city':'上海'}]
print(vec .fit_transform(instances).toarray())


## 文字特征提取
## 字典模式
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
        'UNC played Duke in basketball',
        'Duke lost the basketball game' ]
vectorizer = CountVectorizer()
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

## TF-IDF模式
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
        'The dog ate a sandwich and I ate a sandwich',
        'The wizard transfigured a sandwich' ]
vectorizer = TfidfVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)

## 上面的创建字典太耗内存。
## Hash 技巧将词块用哈希函数来确定它在特征向量的索引位置，可以不创建词典
from sklearn.feature_extraction.text import HashingVectorizer
corpus = [
        'The dog ate a sandwich and I ate a sandwich',
        'The wizard transfigured a sandwich' ]
vectorizer = HashingVectorizer(n_features=6) ## 为了演示方便选了6
print(vectorizer.transform(corpus).todense())