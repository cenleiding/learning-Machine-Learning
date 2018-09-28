# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/9/28
@description: 借助gensim实现词向量空间
"""
import os
import pickle

import numpy as np
import gensim
from gensim.models import word2vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

__all__ =['Gensim_config',
          'preprocess_min_words',
          'MySentence',
          'train_w2v',
          'save_w2v',
          'load_w2v',
          'tsne_plot_embeddings',
          'word_similarity',
          'word_similar_by_word']

class Gensim_config(object):
    def __init__(self):
        self.size = 128                              #词向量维度
        self.window = 5                              #扫描窗口
        self.mim_count = 3                           #最小词频
        self.sg = 0                                  # 0:CBOW 1:skip-gram
        self.hs = 0                                  # 1:hierarchical softmax 0:negative sampling
        self.negative = 10                           # negative sampling 词数
        self.alpha = 0.025                           #学习速率
        self.iter = 5                                #语料库迭代次数
        self.max_vocab_size = None                   #最大词汇数
        
        self.dir_corpus = 'corpus'                   #语料库路径
        self.corpus_type = 0                         # 0:一行型 1:多行型
        self.dir_result = 'result'                   #词向量保存路径
        self.filename_w2id = 'word2id.cpkt'          #w2id表
        self.filename_w2v = 'embedding_matrix.cpkt'  #词向量空间表
        self.filename_model = 'w2v_model'            #gensim训练模型
        self.filename_img = 'tsne.png'               #词向量空间降维图

        self.preprocess_min_word = 1                 #1:提前替换低频词 0:不提前处理最后赋零

def preprocess_min_words(config = Gensim_config()):
    all_word_dict={}
    for fname in os.listdir(config.dir_corpus):
        for line in open(os.path.join(config.dir_corpus, fname),encoding='utf-8'):
            for word in line.split():
                all_word_dict[word] =1 if word not in all_word_dict.keys() else all_word_dict[word]+1
    use_word_dict=[]
    for word,num in all_word_dict.items():
        if num >= config.mim_count:
            use_word_dict.append(word)
    return use_word_dict

class MySentence(object):
    def __init__(self, config = Gensim_config(), use_word_dict = None):
        self.dir = config.dir_corpus
        self.type = config.corpus_type
        self.preprocess_min_word = config.preprocess_min_word
        self.use_word_dict = use_word_dict

    def __iter__(self):
        if self.type == 1:
            for fname in os.listdir(self.dir):
                for line in open(os.path.join(self.dir,fname),encoding='utf-8'):
                    if self.preprocess_min_word == 0:
                        yield line.split()
                    else:
                        newLine = map(lambda x: x if x in self.use_word_dict else 'UNK',line.split())
                        yield  newLine
        else:
            for fname in os.listdir(self.dir):
                for line in word2vec.Text8Corpus(os.path.join(self.dir,fname)):
                    if self.preprocess_min_word == 0:
                        yield line
                    else:
                        newLine = list(map(lambda x: x if x in self.use_word_dict else 'UNK',line))
                        yield  newLine

def train_w2v(sentences = None, config = Gensim_config()):
    model = gensim.models.Word2Vec(sentences=sentences,
                                   size=config.size,
                                   alpha=config.alpha,
                                   window=config.window,
                                   min_count=config.mim_count,
                                   max_vocab_size=config.max_vocab_size,
                                   sg=config.sg,
                                   hs=config.hs,
                                   negative=config.negative,
                                   iter=config.iter)
    return model

def save_w2v(model = None, config = Gensim_config):
    wv = model.wv
    id2word = wv.index2word
    if config.preprocess_min_word == 1:
        word2id = {w: i for i, w in enumerate(id2word)}
        embedding_matrix = np.zeros([len(word2id), config.size])
        for w, i in word2id.items():
            embedding_matrix[i] = wv[w]
    else:
        word2id = {w: i for i, w in enumerate(id2word,1)}
        word2id['UNK']=0
        embedding_matrix = np.zeros([len(word2id), config.size])
        for w, i in word2id.items():
            if w == 'UNK':
                continue
            embedding_matrix[i] = wv[w]

    if not os.path.exists(config.dir_result):
        os.mkdir(config.dir_result)

    model.save(os.path.join(config.dir_result,config.filename_model))

    with open(os.path.join(config.dir_result, config.filename_w2id), 'wb') as wf:
        pickle.dump(word2id, wf)

    embedding_matrix.dump(os.path.join(config.dir_result, config.filename_w2v))

    return id2word,word2id,embedding_matrix

def load_w2v(config = Gensim_config()):

    model = gensim.models.Word2Vec.load(os.path.join(config.dir_result,config.filename_model))

    with open(os.path.join(config.dir_result, config.filename_w2id), 'rb') as rf:
        word2id = pickle.load(rf)

    embedding_matrix = np.load(os.path.join(config.dir_result, config.filename_w2v))

    id2word =['' for i in range(len(word2id.keys()))]
    for word,i in word2id.items():
        id2word[i] = word
    return model,id2word,word2id,embedding_matrix

"""
tsne降维，图像显示,显示100例
"""
def tsne_plot_embeddings(id2word = None, embedding_matrix =None, config = Gensim_config()):
    plt.rcParams['font.sans-serif'] = ['SimHei']      #支持中文
    plt.rcParams['axes.unicode_minus'] = False
    tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
    plot_only = 200
    low_dim_embs = tsne.fit_transform(embedding_matrix[:plot_only,:])
    labels = [id2word[i] for i in range(plot_only)]
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x,y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
    plt.savefig(os.path.join(config.dir_result,config.filename_img))

def word_similarity(word1,word2,model):
    return model.similarity(word1,word2)

def word_similar_by_word(word,model):
    return model.similar_by_word(word,topn=10)