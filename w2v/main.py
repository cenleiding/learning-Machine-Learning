# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/9/26
@description:
"""
from w2v.Participle import Participle
from w2v.W2V_gensim import W2V_gensim

if __name__=='__main__':
    # participle=Participle()
    # participle.File2File()
    # print('分词完毕...')
    # print('训练词向量模型...')
    w2v_gensim=W2V_gensim()
    # w2v_gensim.trainModel()
    w2v_gensim.readModel()
    w2v_gensim.word2vec('1')
    print(w2v_gensim.word_similarity('周芷若','赵敏'))
    similarList=w2v_gensim.word_similar_by_word('张三丰')
    for k in similarList:
        print(k[0], k[1])
    print(w2v_gensim.word2vec(''))