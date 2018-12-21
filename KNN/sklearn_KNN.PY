from sklearn.neighbors import NearestNeighbors
import numpy as np

"""
非监督KNN，用于单纯聚类
"""
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
distances, indices = nbrs.kneighbors(X)
graph = nbrs.kneighbors_graph(X).toarray()
# 展示距离最近的点的索引
print("indices:\n{}".format(indices))
# 展示与距离最近点的距离
print("distance:\n{}".format(distances))
# 展示关系矩阵
print("graph:\n{}".format(graph))

"""
监督KNN，用于分类
"""
X = [[0], [1], [2], [3]]
y = [0, 0, 1, 1]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

# 预测
print(neigh.predict([[1.1]]))

# 各类别概率
print(neigh.predict_proba([[0.9]]))
