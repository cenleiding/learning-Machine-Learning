# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/9/26
@description: 该类用于将中文文本文档进行分词，精确模式
@link: http://github.com/fxsjy/jieba
"""
import jieba
import re

class Participle():

    def __init__(self):
        self.__delete_symbol = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?、~@#￥%……&*（）《》<>”“:：；;]+"

    def Str2Str(self,str):
        str = str
        newLStr = re.sub(self.__delete_symbol, "", str)
        seg_list = jieba.cut(newLStr,cut_all=False)
        return seg_list

    def File2File(self, dir_in='data/origin.txt', dir_out='data/origin_participle.txt'):
        with open(dir_in,'r',encoding='utf-8') as fr,open(dir_out,'w',encoding='utf-8') as fw:
            line = fr.readline()
            while line:
                newLine = re.sub(self.__delete_symbol, "",line)
                newLine = jieba.cut(newLine,cut_all=False)
                newLine = map(lambda s:s+' ',newLine)
                fw.writelines(newLine)
                line = fr.readline()

    def File2Str(self,dir_in='data/origin.txt'):
        with open(dir_in,'r',encoding='UTF-8') as fr:
            text = fr.read()
            text = re.sub(self.__delete_symbol, "",text)
            newText = jieba.cut(text,cut_all=False)
        return newText

    def Str2File(self,str,dir_out='data/origin_participle.txt'):
        with open(dir_out,'w',encoding='UTF-8')as fw:
            str = re.sub(self.__delete_symbol, "",str)
            newStr = jieba.cut(str,cut_all=False)
            newStr = map(lambda s: s + ' ', newStr)
            fw.writelines(newStr)


