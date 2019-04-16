# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/4/16
@description: 数据预处理
"""
from sklearn import preprocessing
import numpy as np

## .scale() 均值为0，方差为1。
## 默认以列为单位处理
## 对于需要距离计算的模型友好，如svm
## 防止特征之间的方差和值的大小相差过大
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.],
                    [0., 1., 3]])
X_scaled = preprocessing.scale(X_train)
print(X_scaled)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))
## .StandardScaler() 能够记录下均值和方差，用于后续处理
scaler = preprocessing.StandardScaler().fit(X_train)
scaler.transform(X_train)
X_test = [[-1., 1., 0.]]
scaler.transform(X_test)

## 函数映射
## .QuantileTransformer 均匀分布映射[0,1]
quantile_transformer = preprocessing.QuantileTransformer()
X_train_trans = quantile_transformer.fit_transform(X_train)
X_test_trans  = quantile_transformer.transform(X_test)

## .PowerTransformer 高斯分布映射
pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))
pt.fit_transform(X_lognormal)

## .normalize()
# 默认行为单位处理
# Normalization主要思想是对每个样本计算其p-范数，然后对该样本中每个元素除以该范数，
# 这样处理的结果是使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1。
# 主要应用于文本分类和聚类中。例如，对于两个TF-IDF向量的l2-norm进行点积，就可以得到这两个向量的余弦相似性。
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
X_normalized = preprocessing.normalize(X, norm='l2')


##########　分类特征编码

## 有序分类
## .OrdinalEncoder()
## 以列为单位
enc = preprocessing.OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'],
     ['female', 'from Europe', 'uses Firefox']]
enc.fit_transform(X) # 设置类别顺序
print(enc.categories_)
print(enc.transform([['female','from US','uses Safari']]))
##  .LabelEncoder() #只能一维
le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
le.transform(["tokyo", "tokyo", "paris"])

## 无序分类
## .OneHotEncoder()
## 返回默认为 scipy.sparse

## 自动学习分类
enc = preprocessing.OneHotEncoder()
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
print(enc.categories_)
print(enc.transform([['female', 'from US', 'uses Safari'],
               ['male', 'from Europe', 'uses Safari']]).toarray())
## 训练集分类不全
enc = preprocessing.OneHotEncoder(handle_unknown='ignore') ## 留出全0的编码用于标示未出现的分类
X = [['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']]
enc.fit(X)
print(enc.transform([['female', 'from Asia', 'uses Chrome']]).toarray())
## 人为指定分类
genders = ['female', 'male']
locations = ['from Africa', 'from Asia', 'from Europe', 'from US']
browsers = ['uses Chrome', 'uses Firefox', 'uses IE', 'uses Safari']
enc = preprocessing.OneHotEncoder(categories=[genders, locations, browsers])
print(enc.fit_transform([['female', 'from Asia', 'uses Chrome']]).toarray())
print(enc.categories_)


## 离散化
## 用于连续属性离散化
## .KBinsDiscretizer 将连续属性分为k个相同宽度的抽屉
X = np.array([[ -3., 5., 15 ],
              [  0., 6., 14 ],
              [  6., 3., 11 ]])
est = preprocessing.KBinsDiscretizer(n_bins=[3, 2, 2], encode='ordinal') #设置每列的K值,encode可以用onehot和ordinal
print(est.fit_transform(X))

## 二值化
## .Binarizer 将特征转化为[0,1]值
X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
binarizer = preprocessing.Binarizer(threshold=1.1) # 设置阈值
print(binarizer.fit_transform(X))


enc = preprocessing.OneHotEncoder()
X = [['male', 'from US', 'uses Safari']]
enc.fit(X)