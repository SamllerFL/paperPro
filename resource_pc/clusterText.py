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


def loadCorpus(file_name, cut_sentence_list, old_sentence_list):
    '''

    :param file_name: 语料库文件路径
    :param cut_sentence_list: 保存去除数字及#后的语句
    :param old_sentence_list: 保存原来的语句
    :return:
    '''
    with open(file_name, encoding='utf8') as f:
        line = f.readline()
        while line:
            code = re.search('[0-9]{3,4}', line)
            word_list = list(jieba.cut(line.strip()))
            if code is not None:
                word_list.remove(code.group())
            if '#' in word_list:
                word_list.remove('#')
            # print(list(word_list))
            cut_sentence_list.append(''.join(word_list))
            old_sentence_list.append(line.strip())
            line = f.readline()


def getCluster(cluster_labels, file_name, line_cluster):
    '''

    :param cluster_labels: 类别标签
    :param file_name: 原语料库路径
    :param line_cluster: 每个类别均保存属于各自的语句列表
    :return:
    '''
    with open(file_name, encoding='utf8') as f:
        line = f.readline()
        for i in cluster_labels:
            if i not in line_cluster:
                line_cluster[i] = []
            line_cluster[i].append(line)
            line = f.readline()


def tfidf_cluster(filename):
    '''

    :param filename: 原语料库路径
    :return:
    '''
    sentence_list = []
    loadCorpus(filename, sentence_list)

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    # 统计每个词语的tf-idf权值
    tfidf_model = vectorizer.fit(sentence_list)
    # print(tfidf_model.vocabulary_)
    tfidf = tfidf_model.transform(sentence_list)
    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    print(word)
    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    tfidf_array = tfidf.toarray()
    # print(tfidf_array)
    # # 查看特征大小
    # print('Features length: ' + str(len(word)))
    # j词在i类文本中
    # TF-IDF 的中文文本 K-means 聚类
    numClass = 8  # 聚类分几簇
    clf = KMeans(n_clusters=numClass, max_iter=10000, init="k-means++", tol=1e-6)  # 这里也可以选择随机初始化init="random"
    # pca = PCA(n_components=20)  # 降维
    # newData = pca.fit_transform(tfidf_array)  # 载入N维
    # cluster = clf.fit(newData)
    cluster = clf.fit(tfidf_array)
    # print(clf.cluster_centers_)
    # print(clf.labels_)
    # print(clf.inertia_)

    line_cluster = {}
    getCluster(clf.labels_, filename, line_cluster)

    for i in range(len(clf.cluster_centers_)):
        print(line_cluster[i][:10])


def word2vec_cluster(filename):
    '''
    建立语句的词向量, 然后用kmeans聚类
    :param filename: 原语料库路径
    :return:
    '''
    cut_sentence_list = []
    old_sentence_list = []
    loadCorpus(filename, cut_sentence_list, old_sentence_list)
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
    numClass = 8
    n_clusters = len(word2vec_model.wv.vocab.keys())
    clf = KMeans(n_clusters=numClass, max_iter=50000, init="k-means++", tol=1e-8)
    s = clf.fit_predict(allword_vector)
    line_cluster = {}

    for i in range(numClass):
        label_i = []
        for j in range(len(s)):
            if s[j] == i:
                label_i.append(old_sentence_list[j])
        line_cluster[i] = label_i
        print('label_' + str(i) + ':' + str(label_i))

    # plt.scatter(pca_vector[:, 0], pca_vector[:, 1], c=s)
    # plt.show()


def doc2vec_cluster(filename):
    '''
    文本聚类
    :param filename: 文本路径, 一行表示一个文本
    :return:
    '''
    def get_dataset(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            docs = f.readlines()
        x_train = []
        old_sentence_list = []
        for i, text in enumerate(docs):
            # code = re.search('[0-9]{3,4}', text)
            # word_list = list(jieba.cut(text.strip()))
            # if code is not None:
            #     word_list.remove(code.group())
            # if '#' in word_list:
            #     word_list.remove('#')
            word_list = ' '.join(jieba.cut(text.strip())).split(' ')  # 保证读入的文件是进行分过词的
            old_sentence_list.append(word_list)
            document = TaggedDocument(word_list, tags=[i])
            x_train.append(document)
        return x_train, old_sentence_list

    x_train, old_sentence_list = get_dataset(filename)
    # 训练文本
    doc2vec_model = Doc2Vec(x_train, min_count=1, window=16, vector_size=120, sample=1e-4, negative=5, workers=8)
    doc2vec_model.train(x_train, total_examples=doc2vec_model.corpus_count, epochs=120)
    doc2vec_model.save('../model/doc2vec_model_total')
    # doc2vec_model = Doc2Vec.load('../model/doc2vec_model_total')

    # 得到文本对应的向量
    allline_vector = [doc2vec_model.infer_vector(line) for line in old_sentence_list]

    # 降维成2维，方便在图中展示
    # pca = PCA(n_components=2)
    # pca_vector = pca.fit_transform(allline_vector)

    # kmeans聚类
    numClass = 8
    clf = KMeans(n_clusters=numClass, max_iter=50000, init="k-means++", tol=1e-8)
    # s = clf.fit_predict(pca_vector)
    s = clf.fit_predict(allline_vector)

    # plt.scatter(pca_vector[:, 0], pca_vector[:, 1], c=s)
    # for i in range(len(s)):  # 给每个点进行标注
    #     plt.annotate(s=s[i], xy=(pca_vector[:, 0][i], pca_vector[:, 1][i]),
    #                  xytext=(pca_vector[:, 0][i] + 0.1, pca_vector[:, 1][i] + 0.1))
    # plt.show()

    # 将字典保存在json文件中
    line_cluster = {}
    for i in range(numClass):
        label_i = []
        for j in range(len(s)):
            if s[j] == i:
                label_i.append(''.join(old_sentence_list[j]))
        line_cluster[i] = label_i
        print('cluster_' + str(i) + ':' + str(label_i))

    # json_str = json.dumps(line_cluster)
    # with open('./file/cluster_file.json', 'w', encoding='utf8') as f:
    #     f.write(json_str)


def process_cluster(cluster_file):
    def lcs(str1, str2):
        dp = [[0]*(len(str2)+1) for i in range(len(str1)+1)]
        for i in range(1, len(str1)+1):
            for j in range(1, len(str2)+1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        i = len(str1) - 1
        j = len(str2) - 1
        str = ''
        while i >= 0 and j >= 0:
            if str1[i] == str2[j]:
                str = str1[i] + str
                j = j - 1
                i = i - 1
            else:
                if dp[i][j-1] > dp[i-1][j]:
                    j = j - 1
                else:
                    i = i - 1
        return str

    with open(cluster_file, 'r', encoding='utf-8') as f:
        line_cluster = json.loads(f.read())
    print(line_cluster)
    re_list = []
    str_list = []
    for key in line_cluster:
        str1 = line_cluster[key][0]
        str2 = line_cluster[key][5]
        str3 = line_cluster[key][6]
        lcs_str = lcs(lcs(str1, str2), str3)
        str_list.append(lcs_str)
        re_list.append('.*'.join(list(jieba.cut(lcs_str))))
    print(str_list)
    print(re_list)

    with open('../file/re_match', 'w', encoding='utf-8') as f:
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


if __name__=='__main__':
    # word2vec_cluster('./file/corpus.txt')
    # tfidf_cluster('./file/corpus.txt')
    doc2vec_cluster('../file/corpus3.txt')
    # process_cluster('../file/cluster_file2.json')
