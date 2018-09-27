# encoding:utf-8

"""
@author:CLD
@file: .py
@time:2018/9/27
@description:
"""
from click._compat import raw_input

from jieba_participle import *

if __name__=='__main__':
    a='aa'
    b=''.join(a)
    a=0
    print(b)
    print(id(a),id(b))
    config = jieba_config()
    print('使用jieba处理文本...自动去除各类中英文符号...字典添加路径：data/userdict.txt')

    method = raw_input('模式选择 { 分词 --[0] , 关键词提取 --[1]}')

    if method == '0':
        method = raw_input('方法选择 { 精确模式 -- [0] ，全模式 -- [1] ，搜索引擎模式 --[2]}:')
        HMM = raw_input('是否开启新词识别 {否 --[0] , 是 --[1]}:')
        contest = raw_input('处理内容 {字符串处理 -- [0] ，文本处理 -- [1]}:')

        if method == '0':
            config.method = 0
            config.cut_all = False
        elif method == '1':
            config.method = 0
            config.cut_all = True
        elif method == '2':
            config.method = 1
        else:
            raise ValueError('输入错误！')

        if HMM == '0':
            config.HMM = False
        elif HMM == '1':
            config.HMM = True
        else:
            raise ValueError('输入错误！')

        if contest == '0':
            while True:
                str = raw_input('输入:')
                newStr = Str2Str(str,config)
                print('输出：'+' '.join(newStr))
        else:
            print('输入文件路径：data/origin.txt   输出文件路径：data/origin_participle.txt')
            file_format = raw_input('输出文件格式 {不换行 -- [0] , 换行 -- [1]}:')
            config.file_format=int(file_format)
            print('文件处理中...')
            File2File(config)
            print('文件处理完毕！')

    elif method == '1':
        print('将提取data/origin.txt文件中 20 个关键词，逆向文件频率文本语料库:data/idf.txt.big,停止词文本语料库:data/stop_words.txt')
        words = Word_analyse()
        print('基于 TF-IDF 算法:'+' '.join(words['extract_tags']))
        print('基于 TextRank 算法:'+' '.join(words['textrank']))