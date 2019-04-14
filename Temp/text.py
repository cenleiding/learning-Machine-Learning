# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2019/2/12
@description:
"""
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import numpy as np
from sklearn.preprocessing import Normalizer
X = np.array(
    [
        [279.09,634.99,54.84,456.4,33582],
        [48.22,71.99,91.82,49.8,4526],
        [143.68,144.84,36.54,143.55,22014],
        [837.18,903.76,55.52,1151.11,56109],
        [235.17,288.43,38.02,396.55,35958],
        [841.96,1625.34,32.26,1287.58,66271],
        [148.79,135.22,49.95,180.42,16536],
        [435.4,947.24,31.71,876.89,65861],
        [201.03,271.87,42.06,304.65,29923],
        [432.16,658.11,17.34,682.03,41895],
        [456.73,867.78,40.65,755.64,44732],
        [209.43,208.87,30.27,186.42,29670],
        [263.39,306.84,36.17,477.76,35808],
        [396.69,676.29,34.34,490.64,30739],
        [177.35,164.39,43.45,200.47,23374]
    ]
)
X = Normalizer().fit_transform(X.T).T
print(X,sep=',')

y=np.array([1,0,0,1,0,1,0,0,0,1,1,0,0,0,0])

label = ['K','YS','shearR','visH','GC']
feature_importances=[0,0,0,0,0]
for i in range(0,1000):
    clf = RandomForestClassifier(n_estimators=20)
    clf.fit(X, y)
    feature_importances=feature_importances+clf.feature_importances_
feature_importances = feature_importances/1000
for i, j in sorted(zip(feature_importances,label)):
    print(i,j)

