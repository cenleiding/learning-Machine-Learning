# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/9/26
@description: 训练词向量空间
"""
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

from gensim.models import word2vec

class W2V_gensim():

    def __init__(self,train_data_dir='data/origin_participle.txt',model_dir='data/w2v_model'):
        self.__train_data_dir=train_data_dir
        self.__model_dir=model_dir

    def trainModel(self):
        sentences=word2vec.Text8Corpus(self.__train_data_dir)
        model=word2vec.Word2Vec(sentences,size=100, alpha=0.025, window=5, min_count=5,
                                sg=0, hs=0, negative=5)
        self.__model=model
        model.save('data/w2v_model')

    def readModel(self):
        self.__model=word2vec.Word2Vec.load(self.__model_dir)

    def word2vec(self,word):
        vec=self.__model[word]
        self.__model.vocabulary
        return vec

    def word_similarity(self,word1,word2):
        return self.__model.similarity(word1,word2)

    def word_similar_by_word(self,word,topn=10):
        return self.__model.similar_by_word(word,topn=topn)


if __name__=='__main__':
    w2v_gensim=W2V_gensim()
    w2v_gensim.trainModel()
    similarList=w2v_gensim.word_similar_by_word('cenleiding')
    for k in similarList:
        print(k[0],k[1])
