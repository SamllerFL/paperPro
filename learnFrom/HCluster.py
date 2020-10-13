# -*- coding:UTF-8 -*-
from numpy import *
import jieba as jb
import time

from sklearn.feature_extraction.text import TfidfVectorizer


def loadDataSet(fileName, fileType='float'):
    dataMat = []
    fr = open(fileName, encoding='utf-8')
    for line in fr.readlines():
        if fileType == 'str':
            curLine = line.strip().split('\t')
        else:
            curLine = map(float, line.strip().split('\t'))
        if curLine != ['']:
            dataMat.append(curLine)
    return dataMat


# 加载停用词，这里主要是排除通用词
def loadStopWords(fileName):
    result = []
    fr = open(fileName, encoding='utf-8')
    for line in fr.readlines():
        result.append(line.strip())
    newWords = []
    for s in result:
        if s not in newWords:
            newWords.append(s)
    newWords.extend([u'（', u'）', '(', ')', '/', '-', '.', '-', '&'])
    return newWords


# 把文本分词并去除停用词，返回数组
def wordsCut(words, stopWordsFile):
    result = jb.cut(words)
    newWords = []
    stopWords = loadStopWords(stopWordsFile)
    for s in result:
        if s not in stopWords:
            newWords.append(s)
    return newWords


# 把样本文件做分词处理，并写文件
def fileCut(fileName, writeFile, stopWordsFile):
    dataMat = []
    fr = open(fileName, encoding='utf-8')
    frW = open(writeFile, 'w', encoding='utf-8')
    for line in fr.readlines():
        curLine = line.strip()
        curLine1 = curLine.lower()  # 把字符串中的英文字母转换成小写
        cutWords = wordsCut(curLine1, stopWordsFile)
        for i in range(len(cutWords)):
            frW.write(cutWords[i])
            frW.write('\t')
        frW.write('\n')
        dataMat.append(''.join(cutWords))
    frW.close()
    return dataMat


# 创建不重复的词条列表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


# 将文本转化为词袋模型
def bagOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec


# 计算所有文本包含的总词数
def wordsCount(dataSet):
    wordsCnt = 0
    for document in dataSet:
        wordsCnt += len(document)
    return wordsCnt


# 计算包含某个词的文本数
def wordInFileCount(word, cutWordList):
    fileCnt = 0
    for i in cutWordList:
        for j in i:
            if word == j:
                fileCnt = fileCnt + 1
            else:
                continue
    return fileCnt


# 计算权值,并存储为txt
def calTFIDF(dataSet, writeFile):
    allWordsCnt = wordsCount(dataSet)  # 所有文本的总词数
    fileCnt = len(dataSet)  # 文本数
    vocabList = createVocabList(dataSet)  # 词条列表
    # tfidfSet = []
    frW = open(writeFile, 'w')
    for line in dataSet:
        wordsBag = bagOfWords2Vec(vocabList, line)  # 每行文本对应的词袋向量
        lineWordsCnt = 0
        for i in range(len(wordsBag)):
            lineWordsCnt += wordsBag[i]  # 计算每个文本中包含的总词数
        tfidfList = [0] * len(vocabList)
        for word in line:
            wordinfileCnt = wordInFileCount(word, dataSet)  # 包含该词的文本数
            wordCnt = wordsBag[vocabList.index(word)]  # 该词在文本中出现的次数
            tf = float(wordCnt) / lineWordsCnt
            idf = math.log(float(fileCnt) / (wordinfileCnt + 1))
            tfidf = tf * idf
            tfidfList[vocabList.index(word)] = tfidf
        frW.write('\t'.join(map(str, tfidfList)))
        frW.write('\n')
        # tfidfSet.append(tfidfList)

    frW.close()
    # return tfidfSet


# 计算余弦距离
def gen_sim(A, B):
    num = float(dot(mat(A), mat(B).T))
    denum = linalg.norm(A) * linalg.norm(B)
    if denum == 0:
        denum = 1
    cosn = num / denum
    sim = 0.5 + 0.5 * cosn  # 余弦值为[-1,1],归一化为[0,1],值越大相似度越大
    sim = 1 - sim  # 将其转化为值越小距离越近
    return sim


def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2)))  # la.norm(vecA-vecB)


def saveResult(clusterAssment, fileName):
    listResult = clusterAssment.tolist()  # 矩阵转换为list
    fr = open(fileName, 'w')
    for i in range(len(listResult)):
        fr.write('\t'.join(map(str, listResult[i])))
        fr.write('\n')
    fr.close()


# 计算簇内两个样本间的最大距离
def diamM(dataSet):
    maxDist = 0
    m = shape(dataSet)[0]
    if m > 1:
        for i in range(m):
            for j in range(i + 1, m):
                dist = gen_sim(dataSet[i, :], dataSet[j, :])
                if dist > maxDist:
                    maxDist = dist
    return maxDist


# 计算两个簇间，样本间的最小距离
def dMin(dataSet1, dataSet2):
    minDist = 1
    m = shape(dataSet1)[0]
    n = shape(dataSet2)[0]
    for i in range(m):
        for j in range(n):
            dist = gen_sim(dataSet1[i, :], dataSet2[j, :])
            if dist < minDist:
                minDist = dist
    return minDist


# 计算簇内样本间的平均距离
def avg(dataSet):
    m = shape(dataSet)[0]
    dist = 0
    avgDist = 0
    if m > 1:
        for i in range(m):
            for j in range(i + 1, m):
                dist += gen_sim(dataSet[i, :], dataSet[j, :])
        avgDist = float(2 * dist) / (m * (m - 1))
    return avgDist


def getTime():
    systime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    return systime


# 计算两个簇的最小距离
def distMin(dataSet1, dataSet2):
    minD = 1
    m = shape(dataSet1)[0]
    n = shape(dataSet2)[0]
    for i in range(m):
        for j in range(n):
            dist = gen_sim(dataSet1[i], dataSet2[j])
            if dist < minD:
                minD = dist
    return minD


# 计算两个簇的最大距离
def distMax(dataSet1, dataSet2):
    maxD = 0
    m = shape(dataSet1)[0]
    n = shape(dataSet2)[0]
    for i in range(m):
        for j in range(n):
            dist = gen_sim(dataSet1[i], dataSet2[j])
            if dist > maxD:
                maxD = dist
    return maxD


# 计算两个簇的评均距离
def distAvg(dataSet1, dataSet2):
    avgD = 0
    sumD = 0
    m = shape(dataSet1)[0]
    n = shape(dataSet2)[0]
    for i in range(m):
        for j in range(n):
            dist = gen_sim(dataSet1[i], dataSet2[j])
            sumD += dist
    avgD = sumD / (m * n)
    return avgD


# 找到距离最近的两个簇
def findMin(M):
    minDist = inf
    m = shape(M)[0]
    for i in range(m):
        for j in range(m):
            if i != j and M[i, j] < minDist:
                minDist = M[i, j]
                minI = i
                minJ = j
    return minI, minJ, minDist


# 计算DI指数，该值越大越好
def DIvalue(dataSet, dataResult, k):
    m = k
    diam = []
    dmin = 1
    DI = 1
    for i in range(m):
        dataSeti = dataSet[nonzero(dataResult[:, 0].A == i)[0], :]
        diam1 = diamM(dataSeti)
        diam.append(diam1)
        for j in range(i + 1, m):
            dataSetj = dataSet[nonzero(dataResult[:, 0].A == j)[0], :]
            dist = dMin(dataSeti, dataSetj)
            if dist < dmin:
                dmin = dist
                I = i
                J = j
    maxDiam = max(diam)
    if maxDiam > 0:
        DI = float(dmin) / max(diam)
    return DI


def centerValue(dataSet):
    m = shape(dataSet)[0]
    n = shape(dataSet)[1]
    sumLine = mat(zeros((1, n)))
    center = mat(zeros((1, n)))
    for i in range(m):
        sumLine += dataSet[i, :]
    if m > 0:
        center = sumLine / float(m)
    return center


# 计算DI指数，该值越大越好
def DBIvalue(dataSet, dataResult, k):
    maxDBI = 0
    sumDBI = 0
    DBI = 0
    for i in range(k):
        for j in range(i + 1, k):
            dataSeti = dataSet[nonzero(dataResult[:, 0].A == i)[0], :]
            dataSetj = dataSet[nonzero(dataResult[:, 0].A == j)[0], :]
            centeri = centerValue(dataSeti)
            centerj = centerValue(dataSetj)
            DBIij = (avg(dataSeti) + avg(dataSetj)) / gen_sim(centeri, centerj)
            if DBIij > maxDBI:
                maxDBI = DBIij
        sumDBI += maxDBI
    if k > 0:
        DBI = sumDBI / float(k)
    return DBI


# 层次聚类算法
def hCluster(dataSet, k, dist, distMeas=distAvg):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m, 1)))
    performMeasure = []
    M = mat(zeros((m, m)))  # 距离矩阵
    # 初始化聚类簇，每个样本作为一个类
    for ii in range(m):
        clusterAssment[ii, 0] = ii

    for i in range(m):
        for j in range(i + 1, m):
            dataSeti = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            dataSetj = dataSet[nonzero(clusterAssment[:, 0].A == j)[0], :]
            M[i, j] = distMeas(dataSeti, dataSetj)
            M[j, i] = M[i, j]
        if mod(i, 10) == 0:
            print(i)
    q = m  # 设置当前聚类个数
    minDist = 0
    # while (q > k):
    while (minDist < dist):
        i, j, minDist = findMin(M)  # 找到距离最小的两个簇
        # 把第j个簇归并到第i个簇
        clusterAssment[nonzero(clusterAssment[:, 0].A == j)[0], 0] = i
        for l in range(j + 1, q):  # 将j之后的簇重新编号
            clusterAssment[nonzero(clusterAssment[:, 0].A == l)[0], 0] = l - 1
        M = delete(M, j, axis=0)
        M = delete(M, j, axis=1)
        for l in range(q - 1):  # 重新计算第i个簇和其他簇直接的距离
            dataSeti = dataSet[nonzero(clusterAssment[:, 0].A == i)[0], :]
            dataSetl = dataSet[nonzero(clusterAssment[:, 0].A == l)[0], :]
            M[i, l] = distMeas(dataSeti, dataSetl)
            M[l, i] = M[i, l]

        # DBI = DBIvalue(dataSet, clusterAssment, q)
        # DI = DIvalue(dataSet, clusterAssment, q)
        DBI = 0
        DI = 0

        performMeasure.append([q - 1, minDist, DBI, DI])

        q = q - 1

        print('---' + getTime() + '---')
        print(u'当前簇的个数是：', q)
        print(u'距离最小的两个簇是第%d个和第%d个,距离是%f,DBI值是%f,DI值是%f' % (
            i, j, minDist, DBI, DI))

    return clusterAssment, mat(performMeasure)


def getTime():
    systime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    return systime


def main(k, dist):
    dir = r'C:\Users\flisn\OneDrive\paperPro\file\\'

    setence_list = fileCut(dir + 'corpus3.txt', dir + 'cut_file3.txt', dir + 'stop_words')
    print(getTime() + '\t' + u'分词成功！')

    # wordSet = loadDataSet(dir + 'cut_file3.txt', fileType='str')
    # calTFIDF(wordSet, dir + 'tf_idf3.txt')
    # print(getTime() + '\t' + u'计算权值成功！')

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
    wordSet = loadDataSet(dir + 'cut_file3.txt', fileType='str')
    vectorizer = TfidfVectorizer(sublinear_tf=True)
    # 统计每个词语的tf-idf权值
    tfidf_model = vectorizer.fit(setence_list)
    # print(tfidf_model.vocabulary_)
    tfidf = tfidf_model.transform(setence_list)
    # 将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
    dataSet = tfidf.toarray()
    print(getTime() + '\t' + u'计算权值成功！')

    # dataSet = mat(loadDataSet(dir + 'tf_idf3.txt'))
    clustAssing, performMeasure = hCluster(dataSet, k, dist)
    print(getTime() + '\t' + u'聚类算法结束！')

    # saveKMeanResult(myCentroids, 'C:\\Python27\\py\\K-Mean\\kMeanCenter.txt')
    saveResult(clustAssing, dir + 'hc_result3.txt')
    saveResult(performMeasure, dir + 'perform_measure3.txt')
    print(getTime() + '\t' + u'计算结果存储成功！')


if __name__ == '__main__':
    main(0, 0.3)
