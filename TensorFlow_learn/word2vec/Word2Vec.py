# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/9/26
@description:skip-Gram 模式
"""
from sklearn.manifold import TSNE
import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import matplotlib.pyplot as plt
import tensorflow as tf


url = 'http://mattmahoney.net/dc/'
vocabulary_size = 5000 #词典数
data_index=0

batch_size = 128 #每次处理128个样本
embedding_size = 128 #向量大小
skip_window = 1 #窗口大小
num_skips = 2 #每个词的样本数

valid_size = 16 #随机抽取的用于验证单词数
valid_window = 100 #从频率最高的100个词中抽取
valid_examples = np.random.choice(valid_window,valid_size,replace=False)

num_sampled = 64 #负样本采用时噪声单词数

num_steps = 100001 #训练次数

dir_word2id = 'word2id.ckpt'
dir_id2vec = 'id2vec.ckpt'
dir_img='tsne.png'

def maybe_download(filename,expected_bytes):
    if not os.path.exists(filename):
        filename,_=urllib.request.urlretrieve(url+filename,filename)
    statinfo = os.stat(filename)
    if statinfo.st_size==expected_bytes:
        print('Found and verified',filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify'+filename+'Can you get it with a browser?')
    return filename

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

#word2id
def build_dataset(words):
    count = [['UNK',-1]]  #词频统计
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = dict()   #词典
    for word,_ in count:
        dictionary[word] = len(dictionary)
    data = list()         #word to id
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index=0
            unk_count+=1
        data.append(index)
    count[0][1]=unk_count
    reverse_dictionary=dict(zip(dictionary.values(),dictionary.keys()))

    return data,count,dictionary,reverse_dictionary

"""
batch:每次用于训练的样本
batch_size:样本数大小
skip_window:窗口大小，单词能最远联系的距离
num_skips:每个单词最多能生成的样本数，不大于skip_window的两倍，且batch_size必须为它的整数倍（保证每个batch中的词都包含了所有样本）
data_index:全局变量，当前单词序号
"""
def generate_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size%num_skips == 0
    assert num_skips<=2*skip_window
    batch = np.ndarray(shape=(batch_size),dtype=np.int32)
    labels = np.ndarray(shape=(batch_size,1),dtype=np.int32)  #写明列为1是因为后面计算nce_loss需要
    span = 2*skip_window+1
    buffer = collections.deque(maxlen=span)  #队列用于存放当前可能用到的所有单词

    for _ in range(span):
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    for i in range(batch_size//num_skips):
        target = skip_window
        targets_to_avoid=[skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0,span-1)       #随机获得一个样本
            targets_to_avoid.append(target)
            batch[i*num_skips+j]=buffer[skip_window]    #目标词
            labels[i*num_skips+j,0]=buffer[target]      #上下文
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    return batch,labels

"""
构建、训练网络
"""
def skip_gram_graph():
    graph = tf.Graph()
    with graph.as_default():
        train_inputs = tf.placeholder(tf.int32,shape=[batch_size])
        train_labels = tf.placeholder(tf.int32,shape=[batch_size,1])
        valid_dataset = tf.constant(valid_examples,dtype=tf.int32)

        with tf.device('/cpu:0'):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1,1)) #词向量空间
            embed = tf.nn.embedding_lookup(embeddings,train_inputs) #从词向量空间中找出对应输入id的输出
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size,embedding_size],stddev=1/math.sqrt(embedding_size))) #每个向量的权重
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))                                         #噪声

        #负采样的过程其实就是优先采词频高的词作为负样本。
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=train_labels,      #真实值
                                             inputs=embed,             #预测值
                                             num_sampled=num_sampled,  #负采样噪声数
                                             num_classes=vocabulary_size))

        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss) #梯度下降

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True)) #向量空间标准化
        normalized_embeddings=embeddings/norm

        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset) #计算验证单词的嵌入向量与词汇表中所有单词的相似性
        similarity = tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True) #嗯..直接以向量乘积近似距离

        init = tf.global_variables_initializer()

    with tf.Session(graph=graph) as session:                     #开始训练网络
        init.run()
        print("Initialized")

        average_loss=0
        for step in range(num_steps):
            batch_inputs,batch_labels=generate_batch(batch_size,num_skips,skip_window)
            feed_dict={train_inputs:batch_inputs,train_labels:batch_labels}
            _,loss_val,embed2=session.run([optimizer,loss,embed],feed_dict=feed_dict)
            average_loss+=loss_val

            if step%2000 == 0:
                if step > 0:
                    average_loss/=2000
                print("Aaverage loss at step",step,":",average_loss)
                average_loss=0

            if step%10000 == 0:                                      #展示与每个验证单词相似的8个单词
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word=reverse_dictionary[valid_examples[i]]
                    top_k=8
                    nearest = (-sim[i,:]).argsort()[1:top_k+1]
                    log_str = "Nearest to %s" % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str="%s %s,"%(log_str,close_word)
                    print(log_str)

            final_embeddings = normalized_embeddings.eval()
    return  final_embeddings

"""
tsne降维，图像显示
"""
def tsne_plot_embeddings(final_embeddings):
    tsne = TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
    plot_only = 100
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    assert low_dim_embs.shape[0] >= len(labels), "more labels than embeddings"
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x,y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
    plt.show()
    plt.savefig(dir_img)

if __name__=='__main__':
    filename = maybe_download('text8.zip',31344016)
    words=read_data(filename)
    print('Data size',len(words))
    data, count, dictionary, reverse_dictionary=build_dataset(words)
    del words #节约内存
    final_embeddings=skip_gram_graph()
    tsne_plot_embeddings(final_embeddings)

