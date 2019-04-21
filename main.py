# -*- coding: utf-8 -*-
from collections import defaultdict
import math
import operator
import os
import copy
from NeuralNetwork import *
import numpy as np
import random

baseTrainFilePath = '/media/patrick/Softwares/PyProject/DianProject/source/20newsbydate/20news-bydate-train/'

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

output_sample = {'alt.atheism' : 0,
              'rec.sport.hockey': 1,
              'comp.graphics': 2,
              'sci.crypt': 3,
              'comp.os.ms-windows.misc': 4,
              'sci.electronics': 5,
              'comp.sys.ibm.pc.hardware': 6,
              'sci.med': 7,
              'comp.sys.mac.hardware': 8,
              'sci.space': 9,
              'comp.windows.x': 10,
              'soc.religion.christian': 11,
              'misc.forsale': 12,
              'talk.politics.guns': 13,
              'rec.autos': 14,
              'talk.politics.mideast': 15,
              'rec.motorcycles': 16,
              'talk.politics.misc': 17,
              'rec.sport.baseball': 18,
              'talk.religion.misc': 19}

saved_word_tf_idf = {}

saved_all_words = {}

saved_word_idf = {}

saved_sample_train = {}

saved_words_needed = {}

def list2string(words_list):
    string_list = []
    for words in words_list:
        string = " ".join(words)
        string_list.append(string)
    return string_list


def file_basic_process(txt):
    txt = txt.lower()
    for ch in "~@#$%^&*()_-+=<`>?/,.:;{}[]|\\!\'\"":
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
            txt = open(baseFilePath + category + '/' + fileName, 'r', errors='replace').read()
            letterVec = file_basic_process(txt)
            words = letterVec.split()
            wordsList.append(words)
        library[category] = wordsList
    return library


def get_words_needed():
    word_tf_idf, word_idf, saved_all_words, saved_sample_train = load_processed_data()
    words_needed = set()
    for category in categories:
        category_len = len(word_tf_idf[category])
        token = [i for i in word_tf_idf[category]][:category_len]
        list1 = [i for i in list(reversed(token))][:int(0.2 * category_len)]
        for tUple in list1:
            words_needed.add(tUple[0])
        print(category + 'has been done!')

    word_idf_bak = copy.deepcopy(word_idf)
    for word_index in word_idf_bak:
        if word_index not in words_needed:
            word_idf.pop(word_index)

    f = open('./data/words_needed.txt', 'w')
    f.write(str(word_idf))
    f.close()


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
    print("excluded words in the doc_frequency have been drop!")

    # 手动去除数字
    doc_frequency_bak = copy.deepcopy(doc_frequency)
    for category in library:
        for key in doc_frequency_bak.get(category).keys():
            if str.isdigit(key):
                doc_frequency[category].pop(key)
    print("nums in the doc_frequency have been drop!")

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
    # warning : list is too slow!!!!!!!
    for word in all_words:
        for essay in essay_set_lists:
            if word in essay:
                word_doc[word] += 1
                continue
        flag = flag + 1
        if flag % 1000 == 0:
            print(str(flag) + " words has been solved.")

    for category in doc_frequency:
        for word in doc_frequency[category]:
            word_idf[word] = math.log(doc_num / (word_doc[word] + 1))
    print("word_idf has been gotten!")

    # 计算每个词的TF * IDF的值
    word_tf_idf = {}
    for category in doc_frequency:
        word_tf_idf_category = {}
        for word in doc_frequency[category]:
            word_tf_idf_category[word] = word_tf[category][word] * word_idf[word]
        word_tf_idf_category = sorted(word_tf_idf_category.items(), key=operator.itemgetter(1), reverse=False)
        word_tf_idf[category] = word_tf_idf_category
    print("word_tf_idf has been gotten!")

    f = open('./data/word_idf_old.txt', 'w')
    f.write(str(word_idf))

    get_words_needed()
    print("word_needed has been gotten!")

    f = open('./data/word_tf_idf.txt', 'w')
    f.write(str(word_tf_idf))
    f = open('./data/word_idf.txt', 'w')
    f.write(str(word_idf))
    f = open('./data/all_words.txt', 'w')
    f.write(str(all_words))
    f.close()
    return word_idf

def preprocess_features():
    library = load_data_set(baseTrainFilePath, categories)  # 加载数据
    shuffle_essays(library)
    features = feature_select(library)  # 所有词的TF-IDF值
    print(features)
    print(len(features))

def load_processed_data():
    f = open('./data/word_tf_idf.txt', 'r')
    temp = f.read()
    saved_word_tf_idf = eval(temp)
    f = open('./data/word_idf.txt', 'r')
    temp = f.read()
    saved_word_idf = eval(temp)
    f = open('./data/all_words.txt', 'r')
    temp = f.read()
    saved_all_words = eval(temp)
    f = open('./data/sample_train.txt', 'r')
    temp = f.read()
    saved_sample_train = eval(temp)
    f = open('./data/words_needed.txt', 'r')
    temp = f.read()
    saved_words_needed = eval(temp)
    f.close()
    print("data has been loaded!")
    return saved_word_tf_idf, saved_word_idf, saved_all_words, saved_sample_train, saved_words_needed

def get_essay_tf(essay, word_idf):
    essay_word_frequency = defaultdict(int)
    essay_tf = {}
    essay_vec = []
    total_words_num = len(essay)
    for word in essay:
        essay_word_frequency[word] = essay_word_frequency[word] + 1
    essay_word_frequency_bak = essay_word_frequency.copy()
    for word in essay_word_frequency_bak:
        if word in excludes:
            essay_word_frequency.pop(word)
    for word in essay_word_frequency:
        essay_tf[word] = essay_word_frequency[word] / total_words_num
    for word in word_idf:
        if word in essay_tf:
            essay_vec.append(essay_tf[word] * word_idf[word])
        else:
            essay_vec.append(0)
    return essay_vec

def shuffle_essays(library):
    essay_list_random = []
    for category in library:
        for essay in library[category]:
            essay_list_random.append({category: essay})
    print(essay_list_random)
    random.shuffle(essay_list_random)
    print(essay_list_random)
    f = open('./data/sample_train.txt', 'w')
    f.write(str(essay_list_random))
    print("essays have been shuffled!")

def bp():
    saved_word_tf_idf, saved_word_idf, saved_all_words, saved_sample_train, saved_words_needed = load_processed_data()
    input_vec = []
    output_vec = []
    for index in range(len(saved_sample_train)):
        input_vec.append(get_essay_tf(list(saved_sample_train[index].values())[0], saved_words_needed))
        output_vec.append(output_sample[list(saved_sample_train[index].keys())[0]])
    data = np.array(input_vec)
    labels = np.array(output_vec)

    s_line = int(0.7 * len(labels))
    valid_X = data[s_line:]
    valid_y = labels[s_line:]

    train_X = data[:s_line]
    train_y = labels[:s_line]

    W1 = np.random.randn(len(saved_words_needed), 5000) / math.sqrt(len(saved_words_needed))
    b1 = np.zeros(5000)
    W2 = np.random.randn(5000, 500) / math.sqrt(5000)
    b2 = np.zeros(500)
    W3 = np.random.randn(500, len(output_sample)) / math.sqrt(500)
    b3 = np.zeros(len(output_sample))
    lr = 0.05
    regu_rate = 0.001
    max_iter = 50000

    fc1 = FC(W1, b1, lr, regu_rate)
    relu1 = Relu()
    fc2 = FC(W2, b2, lr, regu_rate)
    relu2 = Relu()
    fc3 = FC(W3, b3, lr, regu_rate)
    cross_entropy = SparseSoftmaxCrossEntropy()

    for i in range(max_iter):
        h1 = fc1.forward(train_X)
        h2 = relu1.forward(h1)
        h3 = fc2.forward(h2)
        h4 = relu2.forward(h3)
        h5 = fc3.forward(h4)
        loss = cross_entropy.forward(h5, train_y)

        print("iter: {}, loss：{}".format(i + 1, loss))

        grad_h5 = cross_entropy.backprop()
        grad_h4 = fc3.backprop(grad_h5)
        grad_h3 = relu2.backprop(grad_h4)
        grad_h2 = fc2.backprop(grad_h3)
        grad_h1 = relu1.backprop(grad_h2)
        grad_X = fc1.backprop(grad_h1)

        fc2.update()
        fc1.update()

    valid_h1 = fc1.forward(valid_X)
    valid_h2 = relu1.forward(valid_h1)
    valid_h3 = fc2.forward(valid_h2)
    valid_h4 = relu1.forward(valid_h3)
    valid_h5 = fc3.forward(valid_h4)
    valid_predict = np.argmax(valid_h5, 1)

    valid_acc = np.mean(valid_predict == valid_y)
    print('acc: ', valid_acc)


if __name__ == '__main__':
    bp()

