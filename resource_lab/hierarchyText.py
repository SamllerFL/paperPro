# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pylab import mpl
import jieba

mpl.rcParams['font.sans-serif'] = ['SimHei']

'''
    1、加载语料
'''
def get_dataset(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        docs = f.readlines()
    cut_sentence_list = []
    old_sentence_list = []
    for i, text in enumerate(docs):
        # code = re.search('[0-9]{3,4}', text)
        # word_list = list(jieba.cut(text.strip()))
        # if code is not None:
        #     word_list.remove(code.group())
        # if '#' in word_list:
        #     word_list.remove('#')
        old_sentence_list.append(''.join(jieba.cut(text.strip())))
        cut_sentence_list.append(' '.join(jieba.cut(text.strip())))
        # text = text.replace('万悦一段', '')
    return old_sentence_list, cut_sentence_list


# 将文本数据转化为列表
old_sentence_list, cut_sentence_list = get_dataset('../file/corpus3.txt')
# list1 = text.split("\n")
# print(list1)

# 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = TfidfVectorizer(sublinear_tf=True)
# 统计每个词语的tf-idf权值
tfidf_model = vectorizer.fit(old_sentence_list)
# print(tfidf_model.vocabulary_)
tfidf = tfidf_model.transform(old_sentence_list)
tfidf_array = tfidf.toarray()

word = vectorizer.get_feature_names()

# # 提取关键词出现min到max次的关键词
# count_vec = CountVectorizer(min_df=40)
# # 把关键词转化成词篇矩阵
# xx1 = count_vec.fit_transform(cut_sentence_list).toarray()
# # 读取具体关键词
# word = count_vec.get_feature_names()

# xx1 = xx1.T

# 聚类标题为高词频的关键词
titles = word

# 将词篇矩阵转化dataframe
# DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表
df = pd.DataFrame(tfidf_array)
# print(df)
# 距离为corr。距离corr(x,y) 相关系数，用来刻画二维随机变量两个分量间相互关联程度
dist = df.corr()

import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist)  # 使用Ward聚类预先计算的距离定义链接矩阵

fig, ax = plt.subplots(figsize=(10, 6))  # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=titles)

plt.tick_params(
    axis='x',  # 使用 x 坐标轴
    which='both',  # 同时使用主刻度标签（major ticks）和次刻度标签（minor ticks）
    bottom='off',  # 取消底部边缘（bottom edge）标签
    top='off',  # 取消顶部边缘（top edge）标签
    labelbottom='off')

plt.tight_layout()  # 展示紧凑的绘图布局

# 注释语句用来保存图片
plt.savefig('层次聚类2.png', dpi=200)  # 保存图片为 ward_clusters

