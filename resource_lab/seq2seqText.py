import os
import re
from collections import Counter
import numpy as np
from keras.layers import Dense, Input, LSTM
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from seq2seqconf import *


def preprocess_data():
    """语料加载"""
    with open(corpus_path, encoding='utf-8') as f:
        seqs = f.read().lower().split('\n')

    """构建序列和字库"""
    seqs_input, seqs_output = [], []  # 输入、输出序列
    counter_input, counter_output = Counter(), Counter()  # 字库
    for seq in seqs:
        if seq.strip() == '':
            continue
        inputs, outputs = seq.strip().split(',')
        counter_input += Counter(list(inputs))
        counter_output += Counter(list(outputs))
        outputs = chr_start + outputs + chr_end  # 加入起终点
        seqs_input.append(inputs)
        seqs_output.append(outputs)

    # 过滤低频词
    # counter_input = counter_input.most_common(num_classes_input)
    # counter_output = counter_output.most_common(num_classes_output)

    # 加入字符（填充、起点、终点）到字库
    counter_input = [chr_pad] + [i[0] for i in counter_input]
    counter_output = [chr_pad, chr_start, chr_end] + [i[0] for i in counter_output]

    """字符和索引间的映射"""
    chr2id_input = {c: i for i, c in enumerate(counter_input)}
    chr2id_output = {c: i for i, c in enumerate(counter_output)}
    c2i_input = lambda c: chr2id_input.get(c, 0)
    c2i_output = lambda c: chr2id_output.get(c, 0)
    id2chr_output = {i: c for c, i in chr2id_output.items()}
    yield c2i_input, c2i_output, id2chr_output

    """输入层和输出层"""
    # 输入序列
    x_encoder = [[c2i_input(c) for c in chrs if c2i_input(c)] for chrs in seqs_input]
    # 起点 + 输出序列
    x_decoder = [[c2i_output(c) for c in chrs[:-1] if c2i_output(c)] for chrs in seqs_output]
    # 输出序列 + 终点
    y = [[c2i_output(c) for c in chrs[1:] if c2i_output(c)] for chrs in seqs_output]

    # 序列截断或补齐为等长
    x_encoder = pad_sequences(x_encoder, maxlen_input, padding='post', truncating='post')
    x_decoder = pad_sequences(x_decoder, maxlen_output, padding='post', truncating='post')
    y = pad_sequences(y, maxlen_output, padding='post', truncating='post')

    # 独热码
    # x_encoder = to_categorical(x_encoder, num_classes=None)
    # x_decoder = to_categorical(x_decoder, num_classes=None)
    x_encoder = to_categorical(x_encoder, num_classes=num_classes_input)
    x_decoder = to_categorical(x_decoder, num_classes=num_classes_output)
    y = to_categorical(y, num_classes=num_classes_output)
    print('输入维度', x_encoder.shape, x_decoder.shape, '输出维度', y.shape)
    yield x_encoder, x_decoder, y


class Seq2seq:
    def __init__(self):
        self.c2i_input = None
        self.c2i_output = None
        self.id2chr_output = None
        self.x_encoder = None
        self.x_decoder = None
        self.y = None
        self.model = None
        self.model_encoder = None
        self.model_decoder = None
        self.train_model()

    """训练模型"""
    def train_model(self):
        [(self.c2i_input, self.c2i_output, self.id2chr_output), (self.x_encoder, self.x_decoder, self.y)] \
            = list(preprocess_data())

        if os.path.exists(path_hdf5):
            """加载已训练模型"""
            self.model = load_model(path_hdf5)
            self.model_encoder = load_model(path_hdf5_encoder)
            self.model_decoder = load_model(path_hdf5_decoder)
        else:
            """编码模型"""
            # num_classes_input = x_encoder.shape[-1]
            # num_classes_output = x_decoder.shape[-1]

            # '''修改配置文件参数'''
            # with open('seq2seqconf.py', 'r', encoding='utf-8') as f1, open('.newconf', 'w', encoding='utf-8') as f2:
            #     line = f1.readline()
            #     num_re = re.compile('\d+')
            #     while line:
            #         if 'num_classes_input' in line:
            #             line = num_re.sub(str(num_classes_input), line)
            #         if 'num_classes_output' in line:
            #             line = num_re.sub(str(num_classes_output), line)
            #         f2.write(line)
            #         line = f1.readline()
            # os.remove('seq2seqconf.py')  # 删除原来的文件
            # os.rename('.newconf', 'seq2seqconf.py')  # 把新文件的名字改成原来文件的名字

            encoder_input = Input(shape=(None, num_classes_input))  # 编码器输入层
            encoder_lstm = LSTM(units, return_state=True)  # 编码器LSTM层
            _, encoder_h, encoder_c = encoder_lstm(encoder_input)  # 编码器LSTM输出
            self.model_encoder = Model(encoder_input, [encoder_h, encoder_c])  # 【编码模型】

            # 解码器
            decoder_input = Input(shape=(None, num_classes_output))  # 解码器输入层
            decoder_lstm = LSTM(units, return_sequences=True, return_state=True)  # 解码器LSTM层
            decoder_output, _, _ = decoder_lstm(
                decoder_input, initial_state=[encoder_h, encoder_c])  # 解码器LSTM输出
            decoder_dense = Dense(num_classes_output, activation='softmax')  # 解码器softmax层
            decoder_output = decoder_dense(decoder_output)  # 解码器输出

            """训练模型"""
            self.model = Model([encoder_input, decoder_input], decoder_output)  # 【训练模型】
            self.model.compile('adam', 'categorical_crossentropy')
            self.model.fit([self.x_encoder, self.x_decoder], self.y, batchsize, epochs, verbose=2)

            """解码模型"""
            decoder_h_input = Input(shape=(units,))  # 解码器状态输入层h
            decoder_c_input = Input(shape=(units,))  # 解码器状态输入层c
            decoder_output, decoder_h, decoder_c = decoder_lstm(
                decoder_input, initial_state=[decoder_h_input, decoder_c_input])  # 解码器LSTM输出
            decoder_output = decoder_dense(decoder_output)  # 解码器输出
            self.model_decoder = Model([decoder_input, decoder_h_input, decoder_c_input],
                                  [decoder_output, decoder_h, decoder_c])  # 【解码模型】

            # 模型保存
            os.mkdir(prefix)
            plot_model(self.model, path_png, show_shapes=True, show_layer_names=False)
            plot_model(self.model_encoder, path_png_encoder, show_shapes=True, show_layer_names=False)
            plot_model(self.model_decoder, path_png_decoder, show_shapes=True, show_layer_names=False)
            self.model.save(path_hdf5)
            self.model_encoder.save(path_hdf5_encoder)
            self.model_decoder.save(path_hdf5_decoder)

    """序列生成序列"""
    def seq2seq(self, x_encoder_pred):
        h, c = self.model_encoder.predict(x_encoder_pred)
        id_pred = id_start
        seq = ''
        for _ in range(maxlen_output):
            y_pred = to_categorical([[[id_pred]]], num_classes=num_classes_output)
            output, h, c = self.model_decoder.predict([y_pred, h, c])
            id_pred = np.argmax(output[0])
            seq += self.id2chr_output[id_pred]
            if id_pred == id_end:
                break
        return seq[:-1]

    '''翻译入口'''
    def translate(self, input_str):
        chrs = input_str.strip()
        x_encoder_pred = [[self.c2i_input(c) for c in chrs]]
        x_encoder_pred = pad_sequences(x_encoder_pred, maxlen_input, padding='post', truncating='post')
        x_encoder_pred = to_categorical(x_encoder_pred, num_classes_input)
        seq = self.seq2seq(x_encoder_pred)
        return seq


# seq2seq_model = Seq2seq()
# while True:
#     chrs = input('输入：').strip()
#     seq = seq2seq_model.translate(chrs)
#     print('输出：%s\n' % seq)