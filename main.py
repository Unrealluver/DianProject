# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import operator
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

excludes = ['the', 'and', 'to', 'of', 'i', 'a', 'in', 'it', 'that', 'is',
            'you', 'my', 'with', 'not', 'his', 'this', 'but', 'for',
            'me', 's', 'he', 'be', 'as', 'so', 'him', 'your',
            'from', 'on', 'are', 'or', 'have', 'if', 'an']

categories = ['alt.atheism',
              'rec.sport.hockey',
              'comp.graphics',
              'sci.crypt',
              'comp.os.ms-windows.misc',
              'sci.electronics',
              'comp.sys.ibm.pc.hardware',
              'sci.med',
              'comp.sys.mac.hardware',
              'sci.space',
              'comp.windows.x',
              'soc.religion.christian',
              'misc.forsale',
              'talk.politics.guns',
              'rec.autos',
              'talk.politics.mideast',
              'rec.motorcycles',
              'talk.politics.misc',
              'rec.sport.baseball',
              'talk.religion.misc']


def list2string(words_list):
    string_list = []
    for words in words_list:
        string = " ".join(words)
        string_list.append(string)
    return string_list


def file_basic_process(txt):
    txt = txt.lower()
    for ch in "~@#$%^&*()_-+=<>?/,.:;{}[]|\\!\'\"":
        txt = txt.replace(ch, ' ')
    return txt


"""
函数说明:创建数据样本
Returns:
    dataset - 实验样本切分的词条
    classVec - 类别标签向量
"""


def load_data_set(baseFilePath, categories):
    library = {}
    for category in categories:
        fileNames = os.listdir(baseFilePath + category)
        wordsList = []
        for fileName in fileNames:
            # print(filesPath + fileName)
            txt = open(baseFilePath + category + '/' + fileName, 'r', errors='ignore').read()
            letterVec = file_basic_process(txt)
            words = letterVec.split()
            # print(words)
            wordsList.append(words)
        library[category] = wordsList
    # print(wordslist)
    # classVec = [0, 1, 0, 1, 0, 1]  # 类别标签向量，1代表好，0代表不好
    return library


"""
函数说明：特征选择TF-IDF算法
Parameters:
     list_words:词列表
Returns:
     dict_feature_select:特征选择词字典
"""


def feature_select(library):
    # library{dict:category} contains wordsList[list] contains words[list] contains word'string'
    # 总词频统计
    # 依不同的词库产生不同的词频统计
    # doc_frequency{dict:category} contains doc_frequency_category{dict:word} contains num _int
    doc_frequency = defaultdict(dict)
    for category in library:
        doc_frequency_category = defaultdict(int)
        for essay in library[category]:
            for word in essay:
                doc_frequency_category[word] += 1
        doc_frequency[category] = doc_frequency_category
    print("doc_frequency has been gotten!")
    # todo: 手动去常见词
    # doc_frequency_bak = doc_frequency.copy()
    # for doc_frequency_category in doc_frequency:
    #
    #     for key in doc_frequency_bak.keys():
    #         if key in excludes:
    #             doc_frequency.pop(key)

    # 计算每个词的TF值
    # word_tf{dict:category} contains word_tf_category{dict:word} contains num _int
    word_tf = defaultdict(dict)  # 存储每个词的tf值
    for category in library:
        word_tf_category = {}
        for word in doc_frequency[category]:
            word_tf_category[word] = doc_frequency[category][word] / sum(doc_frequency[category].values())
        word_tf[category] = word_tf_category
    print("word_tf has been gotten!")

    # 计算每个词的IDF值
    doc_num = 0
    for category in library:
        doc_num = doc_num + len(library[category])

    word_idf = {}  # 存储每个词的idf值
    word_doc = defaultdict(int)  # 存储包含该词的文档数
    # doc_frequency_category 中包含的word数目是少于原文章的（因为无重复）
    # 但是依然在各个category中有重复的现象
    # 所以我们先做合并工作
    all_words = {}
    for category in doc_frequency:
        all_words.update(doc_frequency[category])
    print("all_words has been gotten, its len is: " + str(len(all_words)))

    # warning : its too slow!!!!!!!
    for word in all_words:
        for category in library:
            for essay in library[category]:
                if word in essay:
                    word_doc[word] += 1
                    break
        print("word: " + word + " has been gotten. " + " num: " + str(word_doc[word]))


    for category in doc_frequency:
        for word in doc_frequency[category]:
            word_idf[word] = math.log(doc_num / (word_doc[word] + 1))
    print("word_idf has been gotten!")

    # 计算每个词的TF*IDF的值
    word_tf_idf = {}
    for category in doc_frequency:
        word_tf_idf_category = {}
        for word in doc_frequency[category]:
            word_tf_idf_category[word] = word_tf[category][word] * word_idf[word]
        word_tf_idf_category = sorted(word_tf_idf_category.items(), key=operator.itemgetter(1), reverse=True)
        word_tf_idf[category] = word_tf_idf_category
    print("word_tf_idf has been gotten!")
    # 对字典按值由大到小排序
    # dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return word_tf_idf


if __name__ == '__main__':
    baseFilePath = '/media/patrick/Softwares/PyProject/DianProject/source/20newsbydate/20news-bydate-train/'
    library = load_data_set(baseFilePath, categories)  # 加载数据
    features = feature_select(library)  # 所有词的TF-IDF值
    print(features)
    print(len(features))
    # tfidf2 = TfidfVectorizer()
    # print(list2string(wordsList))
    # re = tfidf2.fit_transform(list2string(wordsList))
