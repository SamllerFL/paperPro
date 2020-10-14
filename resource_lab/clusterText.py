import jieba
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.word2vec import LineSentence
from sklearn.cluster import KMeans
from gensim.models.doc2vec import TaggedDocument
import matplotlib.pyplot as plt
import json
import re


# 加载停用词，这里主要是排除通用词
def load_stopwords(stopwords_file):
    stopwords_list = []
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            jieba.add_word(line.strip())
            stopwords_list.append(line.strip())
    return stopwords_list


# 把文本分词并去除停用词，返回数组
def cut_stopwords(line, stopwords_file):
    without_stopwords = []
    stopwords_list = load_stopwords(stopwords_file)
    for word in list(jieba.cut(line.strip())):
        if word not in stopwords_list:
            without_stopwords.append(word)
    return without_stopwords


def load_corpus(corpus_file, stopwords_file):
    '''
    导入语料库并预处理
    :param corpus_file: 原始语料文本
    :param stopwords_file: 停用词
    :return: 原始语料列表、去除数字及停用词之后的列表
    '''
    # 原始文本
    old_sentence_list = []
    # 除去数字和停用词的文本
    process_sentence_list = []
    with open(corpus_file, encoding='utf8') as f:
        line = f.readline()
        while line:
            old_sentence_list.append(line.strip())
            # code = re.search('[0-9]{3,4}#', line)
            # if code is not None:
            #     num_re = re.compile('[0-9]{3,4}#')
            #     line = num_re.sub('', line)
            word_list = cut_stopwords(line, stopwords_file)
            process_sentence_list.append(''.join(word_list))
            line = f.readline()
    return old_sentence_list, process_sentence_list


def kmeans_cluster(num_class, text_mat, old_sentence_list):
    '''
    用kmeans聚类
    :param num_class: 分类数
    :param text_array: 矩阵, tfidf或者word2vec
    :param old_sentence_list: 原文本列表
    :return:
    '''
    clf = KMeans(n_clusters=num_class, max_iter=60000, init="k-means++", tol=1e-8)
    s = clf.fit_predict(text_mat)
    line_cluster = {}

    for i in range(num_class):
        label_i = []
        for j in range(len(s)):
            if s[j] == i:
                if type(old_sentence_list) == 'str':
                    label_i.append(old_sentence_list[j])
                else:
                    label_i.append(''.join(old_sentence_list[j]))
        line_cluster[i] = label_i
        print('label_' + str(i) + ':' + str(label_i))
    return line_cluster


def tfidf_cluster(corpus_file, stopwords_file):
    '''
    tf-idf聚类
    :param corpus_file: 原始语料库
    :param stopwords_file: 停用词
    :return:
    '''

    old_sentence_list, cut_sentence_list = load_corpus(corpus_file, stopwords_file)

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    # 统计每个词语的tf-idf权值
    tfidf_model = vectorizer.fit(cut_sentence_list)
    # print(tfidf_model.vocabulary_)
    tfidf = tfidf_model.transform(cut_sentence_list)
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    print(word)
    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    tfidf_array = tfidf.toarray()
    # print(tfidf_array)
    # # 查看特征大小
    # print('Features length: ' + str(len(word)))

    # TF-IDF 的中文文本 K-means 聚类
    num_class = 8  # 聚类分几簇
    # pca = PCA(n_components=20)  # 降维
    # newData = pca.fit_transform(tfidf_array)  # 载入N维

    line_cluster = kmeans_cluster(num_class, tfidf_array, old_sentence_list)


def word2vec_cluster(corpus_file, stopwords_file):
    '''
    建立语句的词向量, 然后用kmeans聚类
    :param filename: 原语料库路径
    :return:
    '''
    old_sentence_list, cut_sentence_list = load_corpus(corpus_file, stopwords_file)
    new_sentence_list = [[line] for line in old_sentence_list]
    word2vec_model = Word2Vec(new_sentence_list, window=5, min_count=1, workers=4)
    word2vec_model.wv.save_word2vec_format('./model/word2vec_model.txt', binary=False)
    print(word2vec_model.wv.vocab.keys())
    # 获取词对于的词向量
    keyword_vector = []
    for key in word2vec_model.wv.vocab.keys():
        keyword_vector.append(word2vec_model[key])

    allword_vector = word2vec_model[old_sentence_list]

    # 降维成2维，方便在图中展示
    pca = PCA(n_components=2)
    pca_vector = pca.fit_transform(allword_vector)

    # kmeans聚类
    num_class = 8
    line_cluster = kmeans_cluster(num_class, allword_vector, old_sentence_list)

    # plt.scatter(pca_vector[:, 0], pca_vector[:, 1], c=s)
    # plt.show()


def doc2vec_cluster(corpus_file, stopwords_file, cluster_file):
    '''
    文本聚类
    :param filename: 文本路径, 一行表示一个文本
    :return:
    '''
    old_sentence_list, cut_sentence_list = load_corpus(corpus_file, stopwords_file)
    old_cut_sentence_list = []
    x_train = []
    for i, line in enumerate(cut_sentence_list):
        old_cut_sentence_list.append(' '.join(jieba.cut(line.strip())).split(' '))
        word_list = ' '.join(jieba.cut(line.strip())).split(' ')  # 保证读入的文件是进行分过词的
        document = TaggedDocument(word_list, tags=[i])
        x_train.append(document)

    # 训练文本
    doc2vec_model = Doc2Vec(x_train, min_count=1, window=15, vector_size=120, sample=1e-4, negative=5, workers=8)
    doc2vec_model.train(x_train, total_examples=doc2vec_model.corpus_count, epochs=120)
    doc2vec_model.save('../model/doc2vec_model_total2.1')
    # doc2vec_model = Doc2Vec.load('../model/doc2vec_model_total2.1')

    # 得到文本对应的向量
    allline_vector = [doc2vec_model.infer_vector(line) for line in old_cut_sentence_list]

    # 降维成2维，方便在图中展示
    # pca = PCA(n_components=2)
    # pca_vector = pca.fit_transform(allline_vector)

    # kmeans聚类
    num_class = 8
    line_cluster = kmeans_cluster(num_class, allline_vector, old_sentence_list)

    # plt.scatter(pca_vector[:, 0], pca_vector[:, 1], c=s)
    # for i in range(len(s)):  # 给每个点进行标注
    #     plt.annotate(s=s[i], xy=(pca_vector[:, 0][i], pca_vector[:, 1][i]),
    #                  xytext=(pca_vector[:, 0][i] + 0.1, pca_vector[:, 1][i] + 0.1))
    # plt.show()

    json_str = json.dumps(line_cluster)
    with open(cluster_file, 'w', encoding='utf-8') as f:
        f.write(json_str)


def process_cluster(cluster_file, stopwords_file, re_match_file):
    def lcs(str1, str2):
        dp = [[0]*(len(str2)+1) for _ in range(len(str1)+1)]
        for i in range(1, len(str1)+1):
            for j in range(1, len(str2)+1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        i = len(str1)
        j = len(str2)
        str = ''
        while i >= 1 and j >= 1:
            if str1[i-1] == str2[j-1]:
                str = str1[i-1] + str
                j = j - 1
                i = i - 1
            else:
                if dp[i][j-1] > dp[i-1][j]:
                    j = j - 1
                else:
                    i = i - 1
        return str

    def match_ratio(re_str, str_list):
        count = 0
        for str in str_list:
            if re.search(re_str, str) is not None:
                count += 1
        return count/len(str_list)

    with open(cluster_file, 'r', encoding='utf-8') as f:
        line_cluster = json.loads(f.read())
    # print(line_cluster)
    re_list = []
    str_list = []
    for key in line_cluster:
        # temp_re = temp_str = ''
        # for i, line in enumerate(line_cluster[key]):
        #     str1 = cut_stopwords(line, stopwords_file)
        #     ratio = 0
        #     for line2 in line_cluster[key][i+1:]:
        #         str2 = cut_stopwords(line2, stopwords_file)
        #         lcs_str = lcs(str1, str2)
        #         re_str = '.*'.join(list(jieba.cut(lcs_str)))
        #         temp_ratio = match_ratio(re_str, line_cluster[key])
        #         if temp_ratio > ratio:
        #             ratio = temp_ratio
        #             temp_re = re_str
        #             temp_str = lcs_str
        # re_list.append(temp_re)
        # str_list.append(temp_str)
        print(line_cluster[key])
        num_re = re.compile('[0-9]{3,4}')
        str1 = ''.join(cut_stopwords(num_re.sub('', line_cluster[key][0]), stopwords_file))
        str2 = ''.join(cut_stopwords(num_re.sub('', line_cluster[key][7]), stopwords_file))
        # str3 = ''.join(cut_stopwords(num_re.sub('', line_cluster[key][0]), stopwords_file))
        # lcs_str = lcs(lcs(str1, str2), str3)
        lcs_str = lcs(str1, str2)
        str_list.append(lcs_str)
        re_list.append('.*'.join(list(jieba.cut(lcs_str))))
    print(str_list)
    print(re_list)

    with open(re_match_file, 'w', encoding='utf-8') as f:
        for i, temp_re in enumerate(re_list):
            f.write(temp_re+','+str_list[i]+'\n')

    re_ratio = []
    for i, key in enumerate(line_cluster):
        list_i = line_cluster[key]
        count = 0
        for sentence in list_i:
            if re.search(re_list[i], sentence) is not None:
                count += 1
        re_ratio.append(count/len(list_i))

    print(re_ratio)


if __name__ == '__main__':
    load_stopwords('../file/stop_words')
    # word2vec_cluster('../file/corpus.txt')
    # tfidf_cluster('../file/corpus3.txt', '../file/stop_words')
    # doc2vec_cluster('../file/corpus3.txt', '../file/stop_words', '../file/cluster_file3.1.json')
    process_cluster('../file/cluster_file3.1.json', '../file/stop_words', '../file/re_match3.1')
