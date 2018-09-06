# coding:utf-8

"""
@author : CLD
@time:2018/9/619:09
@description: 眼镜问题尝试
"""
from DecisionTree import trees,treePlotter

if __name__=='__main__':
    fr=open('lenses.txt')
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels=['age','prescript','astigmatic','tearRate']
    lensesTree=trees.createTree(lenses,lensesLabels)
    treePlotter.createPlot(lensesTree)

