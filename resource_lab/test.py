import jieba
import re, os
from keras.utils import to_categorical
import testconf
from stanfordcorenlp import StanfordCoreNLP
import nltk
nltk.download('punkt')
# num_classes_input = 12
# num_classes_output = 12
# with open('testconf.py', 'r', encoding='utf-8') as f1, open('.newconf', 'w', encoding='utf-8') as f2:
#     line = f1.readline()
#     num_re = re.compile('\d+')
#     while line:
#         if 'num_classes_input' in line:
#             line = num_re.sub(str(num_classes_input), line)
#         if 'num_classes_output' in line:
#             line = num_re.sub(str(num_classes_output), line)
#         f2.write(line)
#         line = f1.readline()
# os.remove('testconf.py')# 删除原来的文件
# os.rename('.newconf', 'testconf.py')#把新文件的名字改成原来文件的名字

# print(len('由“试验”位置摇至“工作”位置'))
# print(len('operateKnifeSwitch//close'))
# print(list(jieba.cut('拉开压变开关')))
# print(list(jieba.cut('检查压变开关三相确在分闸位置')))
# print(list(jieba.cut('将压变刀闸手车由“工作”位置摇至“试验”位置')))
# print(list(jieba.cut('检查压变刀闸手车确在“试验”位置')))
# print(list(jieba.cut('将压变刀闸手车由“试验”位置摇至“工作”位置')))
# print(list(jieba.cut('检查压变刀闸手车确在“工作”位置')))
# print(list(jieba.cut('合上压变刀闸')))
# print(list(jieba.cut('检查压变刀闸三相确已合上')))

# model = StanfordCoreNLP(lang='zh')
# str = model.ner("检查万悦一段4216#开关手车确在“试验”位置")
# print(str)

text = "检查万悦一段4216#开关手车确在“试验”位置"
tokenized = nltk.word_tokenize(text)     # 分词
tagged = nltk.pos_tag(tokenized)         # 词性标注
chunked = nltk.ne_chunk(tagged)          # 命名实体识别
print(chunked)