# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/9/26
@description:
"""
import argparse

from click._compat import raw_input

from gensim_w2v import *

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='配置修改')

    parser.add_argument('--mode',default='train',choices=['train','test'],help='train:训练模型  test:测试模型')

    parser.add_argument('--size',type=int,default=128,help='词向量维度')
    parser.add_argument('--window',type=int,default=5,help='扫描窗口')
    parser.add_argument('--mim_count',type=int,default=3,help='最小词频')
    parser.add_argument('--sg',type=int,default=0,choices=[0, 1],help='0:CBOW 1:skip-gram')
    parser.add_argument('--hs',type=int,default=0,choices=[0, 1],help='1:hierarchical softmax 0:negative sampling')
    parser.add_argument('--negative',type=int,default=10,help='negative sampling 词数')
    parser.add_argument('--alpha',type=float,default=0.025,help='学习速率')
    parser.add_argument('--iter',type=int,default=5,help='语料库迭代次数')
    parser.add_argument('--max_vocab_size',type=int,default=None,help='词向量维度')
    parser.add_argument('--dir_corpus',default='corpus',help='语料库路径')
    parser.add_argument('--corpus_type',type=int,default=0,choices=[0, 1],help='语料类型 0:一行型 1:多行型')
    parser.add_argument('--dir_result',default='result',help='词向量保存路径')
    parser.add_argument('--filename_w2id',default='word2id.cpkt',help='w2id表')
    parser.add_argument('--filename_w2v',default='embedding_matrix.cpkt',help='词向量空间表')
    parser.add_argument('--filename_model',default='w2v_model',help='gensim训练模型')
    parser.add_argument('--filename_img',default='tsne.png',help='词向量空间降维图')
    parser.add_argument('--preprocess_min_word',default=1,choices=[0, 1],help='1:提前替换低频词 0:不提前处理最后赋零')

    args = parser.parse_args()

    gensim_config = Gensim_config()
    for arg in vars(args):
        setattr(gensim_config,arg,getattr(args,arg))
    use_word_dict=preprocess_min_words(gensim_config)

    if args.mode == 'train':
        print('开始读取语料库,路径：'+args.dir_corpus)
        mySentence = MySentence(config=gensim_config,use_word_dict=use_word_dict)

        print('开始模型训练...')
        model = train_w2v(mySentence,gensim_config)
        print('模型训练完成...')
        print('模型结果保存路径：'+args.dir_result)
        id2word, word2id, embedding_matrix = save_w2v(model,gensim_config)
        tsne_plot_embeddings(id2word,embedding_matrix,gensim_config)
    else:
        print('模型读取路径：'+args.dir_result)
        model, id2word, word2id, embedding_matrix=load_w2v(gensim_config)
        print('模型视图生成...')
        tsne_plot_embeddings(id2word,embedding_matrix,gensim_config)
        method = raw_input('测试模式{ 两词相似度--[0] , 相似词列表--[1]}:')
        if method =='0':
            while True :
                word1 = raw_input('词1 ：')
                word2 = raw_input('词2 ：')
                print(word_similarity(word1,word2,model))
        else:
            while True :
                word =raw_input('目标词 ：')
                similarList = word_similar_by_word(word,model)
                for k in similarList:
                    print(k[0], k[1])



