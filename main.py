# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import operator
import os
import copy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

excludes = ['very', 'ourselves', 'am', 'doesn', 'through', 'me', 'against', 'up', 'just', 'her', 'ours',
            'couldn', 'because', 'is', 'isn', 'it', 'only', 'in', 'such', 'too', 'mustn', 'under', 'their',
            'if', 'to', 'my', 'himself', 'after', 'why', 'while', 'can', 'each', 'itself', 'his', 'all', 'once',
            'herself', 'more', 'our', 'they', 'hasn', 'on', 'ma', 'them', 'its', 'where', 'did', 'll', 'you',
            'didn', 'nor', 'as', 'now', 'before', 'those', 'yours', 'from', 'who', 'was', 'm', 'been', 'will',
            'into', 'same', 'how', 'some', 'of', 'out', 'with', 's', 'being', 't', 'mightn', 'she', 'again', 'be',
            'by', 'shan', 'have', 'yourselves', 'needn', 'and', 'are', 'o', 'these', 'further', 'most', 'yourself',
            'having', 'aren', 'here', 'he', 'were', 'but', 'this', 'myself', 'own', 'we', 'so', 'i', 'does', 'both',
            'when', 'between', 'd', 'had', 'the', 'y', 'has', 'down', 'off', 'than', 'haven', 'whom', 'wouldn',
            'should', 've', 'over', 'themselves', 'few', 'then', 'hadn', 'what', 'until', 'won', 'no', 'about',
            'any', 'that', 'for', 'shouldn', 'don', 'do', 'there', 'doing', 'an', 'or', 'ain', 'hers', 'wasn',
            'weren', 'above', 'a', 'at', 'your', 'theirs', 'below', 'other', 'not', 're', 'him', 'during', 'which']

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

saved_word_tf_idf = {}

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

    # 手动去停用词
    doc_frequency_bak = copy.deepcopy(doc_frequency)
    for category in library:
        for key in doc_frequency_bak.get(category).keys():
            if key in excludes:
                doc_frequency[category].pop(key)

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

    # essay_list转set
    essay_set_lists = []
    for category in library:
        for essay in library[category]:
            essay_set_lists.append(set(essay))
    flag = 0
    # warning : its too slow!!!!!!!
    for word in all_words:
        for essay in essay_set_lists:
            if word in essay:
                word_doc[word] += 1
                continue
        flag = flag + 1
        if flag % 1000 == 0:
            print(str(flag) + " words has been solved.")
            # print("essay has been done, word_doc: " + str(word_doc[word]))
        # print("word: " + word + " has been gotten. " + " num: " + str(word_doc[word]))


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

    # np.save('./data/word_tf_idf.npy', word_tf_idf)
    f = open('./data/word_tf_idf.txt', 'w')
    f.write(str(word_tf_idf))
    f.close()
    # 对字典按值由大到小排序
    # dict_feature_select = sorted(word_tf_idf.items(), key=operator.itemgetter(1), reverse=True)
    return word_tf_idf


if __name__ == '__main__':
    # baseFilePath = '/media/patrick/Softwares/PyProject/DianProject/source/20newsbydate/20news-bydate-train/'
    # library = load_data_set(baseFilePath, categories)  # 加载数据
    # features = feature_select(library)  # 所有词的TF-IDF值
    # print(features)
    # print(len(features))

    # tf_example
    # tfidf2 = TfidfVectorizer()
    # print(list2string(wordsList))
    # re = tfidf2.fit_transform(list2string(wordsList))


    f = open('./data/word_tf_idf.txt', 'r')
    temp  =f.read()
    saved_word_tf_idf = eval(temp)
    f.close()

    # saved_word_tf_idf = np.load('./data/word_tf_idf.npy')
    print("word_tf_idf has been read! type of word_tf_idf is " + str(type(saved_word_tf_idf)))
