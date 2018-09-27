# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/9/27
@description: 结巴分词小工具
"""
import argparse

import jieba
import jieba.analyse
import re

__all__ = ['jieba_config','Str2Str','File2File','Word_analyse']

class jieba_config(object):
    def __init__(self):
        self.delete_symbol = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）《》<>”“:：；;]+"
        self.dir_in = 'data/origin.txt'
        self.dir_out = 'data/origin_participle.txt'
        self.dir_dict = 'data/corpus/userdict.txt'
        self.dir_idf = 'data/corpus/idf.txt.big'
        self.dir_stop_words = 'data/corpus/stop_words.txt'
        self.file_format = 0  # 0:不换行 1:换行
        self.method = 0       # 0:cut  1:cut_for_search
        self.cut_all = False  # False：精确模式  True:全模式
        self.HMM = False      # False: 关闭新词识别 True:开启新词识别
        self.topk=20          # 关键词提取数

def Str2Str(str,config=jieba_config()):
    jieba.load_userdict(config.dir_dict)
    str = re.sub(config.delete_symbol, "", str)
    if config.method==0:
        seg_list=jieba.cut(str,cut_all=config.cut_all,HMM=config.HMM)
    else:
        seg_list=jieba.cut_for_search(str,HMM=config.HMM)
    return seg_list

def File2File(config=jieba_config()):
    with open(config.dir_in,'r',encoding='utf-8') as fr,open(config.dir_out,'w',encoding='utf-8') as fw:
        for line in fr:
            newLine=Str2Str(line)
            fw.writelines(' '.join(newLine))
            if config.file_format == 1:
                fw.write('\n')

"""
关键词提取
"""
def Word_analyse(config=jieba_config()):
    jieba.load_userdict(config.dir_dict)
    jieba.analyse.set_idf_path(config.dir_idf)
    jieba.analyse.set_stop_words(config.dir_stop_words)
    with open(config.dir_in, 'r', encoding='utf-8') as fr:
        sentence = fr.read()
        sentence = re.sub(config.delete_symbol, "", sentence)
    textrank = jieba.analyse.textrank(''.join(sentence), topK=20, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
    extract_tags = jieba.analyse.extract_tags(''.join(sentence), topK=config.topk, withWeight=False, allowPOS=())
    words={}
    words['extract_tags']=extract_tags
    words['textrank']=textrank
    return words



