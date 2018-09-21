# 机器学习绪论

就当被单词了  :smile:

## 基本术语

**机器学习（machine learning）** 的目标是通过 **学习\训练算法（learning\training algorithm）**获得数据的 **模型（model）** ,这个模型对应了数据的某种规则因此也叫**假设（hypothesis）**。

模型用于**预测（prediction）**数据，其适用于新数据的能力叫**泛化（generalization）能力** 。

获得模型的过程是一个归纳过程，也称为**归纳学习（inductive learning）**

学习过程可以看做一个在所有假设组成的空间中搜索一个**匹配（fit）** 训练集的假设，但由于训练集有限，往往会获得多个匹配的假设，这个假设集合称为**版本空间（version space）** 。为了获得“最好”的假设，需要对某类型的假设有所偏好，称为**归纳偏好（inductive bias）** 。

---

数据的集合称为**数据集（data set）** ，可以分为**训练集（training set）**、**测试集（testing set）**、**交叉验证集（cross validation set）**

数据集中每一条记录称为 **示例（instance）**或 **样本（sample）** 

如果记录中含有指示结果的 **标签、标记（label）** ，那这个样本称为 **样例（example）**

每一个事项称为**属性（attribute）**或 **特征（feature）**  ，其数量称为样本的**维数（dimensionality）**

所有属性形成的多维空间称为**属性空间（attribute space）** 或 **样本空间（sample space）** 或 **输入空间**

所有标签的集合称为**标记空间（label space）** 或 **输出空间**

在这个空间中每个样本都有自己的坐标，和原点组成一个坐标向量，称为**特征向量（feature vector）**

---

根据是否有标记信息，机器学习可以分为有标记的**监督学习（supervised learning）**和无标记的**无监督学习（unsupervised learning）**。监督学习有分类、回归。非监督学习有**聚类（clustering）**。

根据预测值的类型，机器学习可以分为离散值的**分类（classification）**任务和连续值的**回归（regression）**任务

分类问题又可以分为**二分类（binary classification）** 和 **多分类（multi-class classification）**







## 拓展术语

**奥卡姆剃刀（Occam's razor）** ： 若有多个假设和观察一致，则选择最简单的那个。

**没有免费的午餐 NFL（No Free Lunch Theorem）定理** ：在所有问题出现的机会相同、或所有问题同等重要的情况下， 所有的算法，它们的期望性能都是相同的 ！！ 因此脱离具体问题空谈什么算法更好毫无意义，算法优劣必须针对具体的问题。