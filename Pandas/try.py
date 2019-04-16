# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/4/11
@description:
"""
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import preprocessing


s= pd.Series([1,3,5,np.nan,6])

dates = pd.date_range('20100101',periods=6)
# print(dates)
df = pd.DataFrame(np.random.rand(6,4),index=dates,columns=list('ABCD'))
# print(df.describe())
# print(df.sort_index(axis=1,ascending=False))
# print(df.sort_values(by='A'))

#select
# print(df['A'])
# print(df[0:3])
# print(df['A'][0:3])
# print(df.loc['20100104':])
# print(df.loc[:,['A','B']])
# print(df.iloc[3:5,1:3])
# print(df.iat[0,1])
# print(df[df.A>0.3])

df['E'] = pd.Series([1,2,3,4,np.nan,5],index=dates)

# print(df.dropna(how='any'))
# print(df.fillna(666))
# print(pd.isna(df))

# print(df.mean())
# print(df.mean(1))

# print(df.apply(np.cumsum))
# print(df.apply(lambda x:x.max()-x.min()))

#Merge
df = pd.DataFrame(data=np.random.rand(10,4))
# print(df)
# pieces = [df[:3],df[:3],df[7:]]
# print(pd.concat(pieces))

left = pd.DataFrame({'key': ['foo', 'foo'], 'lval': [1, 2]})
right = pd.DataFrame({'key': ['foo', 'foo'], 'rval': [4, 5]})
# print(pd.merge(left,right))
print(pd.merge(left,right,on='key'))

enc = preprocessing.OneHotEncoder()
X = np.array([['male', 'from US', 'uses Safari'], ['female', 'from Europe', 'uses Firefox']])
x1 = enc.fit_transform(X[:,0].reshape(-1,1))
x2 = enc.fit_transform(X[:,1].reshape(-1,1))
sp = sparse.hstack((x1,x2))
df = pd.DataFrame({'A':[1,2],'B':[3,4]})
sp = sparse.hstack((df['B'].values.reshape(-1,1),sp))
print(sp.toarray())
