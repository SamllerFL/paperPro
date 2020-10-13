import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import copy

vocab_size = 256  # 假设词典大小为 256
target_vocab_size = vocab_size
LR = 0.006

inSize = 10
# outSize = 20  假设输入输出句子一样长
buckets = [(inSize, inSize)]  # 设置一个桶，主要是为了给model_with_buckets函数用
batch_size = 1
input_data = np.arange(inSize)
target_data = copy.deepcopy(input_data)
np.random.shuffle(target_data)
target_weights = ([1.0] * inSize + [0.0] * 0)


class Seq2Seq(object):
    def __init__(self, source_vocab_size, target_vocab_size, buckets, size):
        self.encoder_size, self.decoder_size = buckets[0]  # 因为只有一个桶，索引为0即可
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        cell = tf.contrib.rnn.BasicLSTMCell(size)
        cell = tf.contrib.rnn.MultiRNNCell([cell])

        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
            return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs, decoder_inputs, cell,
                num_encoder_symbols=source_vocab_size,
                num_decoder_symbols=target_vocab_size,
                embedding_size=size,
                feed_previous=do_decode)

        # computational graph
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []

        for i in range(self.encoder_size):
            self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='encoder{0}'.format(i)))

        for i in range(self.decoder_size):
            self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name='decoder{0}'.format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None], name='weights{0}'.format(i)))

        targets = [self.decoder_inputs[i] for i in range(len(self.decoder_inputs))]  # - 1

        # 使用seq2seq，输出维度为seq_length x batch_size x dict_size
        self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets,
            self.target_weights, buckets,
            lambda x, y: seq2seq_f(x, y, False))

        self.getPoints = tf.argmax(self.outputs[0], axis=2)  # 通过argmax，得到字典中具体的值，因为i只有一个批次，所以取0即可
        self.trainOp = tf.train.AdamOptimizer(LR).minimize(self.losses[0])

    def step(self, session, encoder_inputs, decoder_inputs, target_weights):
        input_feed = {}
        for l in range(self.encoder_size):
            input_feed[self.encoder_inputs[l].name] = [encoder_inputs[l]]
        for l in range(self.decoder_size):
            input_feed[self.decoder_inputs[l].name] = [decoder_inputs[l]]
            input_feed[self.target_weights[l].name] = [target_weights[l]]

        output_feed = [self.losses[0], self.getPoints, self.trainOp]
        outputs = session.run(output_feed, input_feed)
        return outputs[0], outputs[1]


# 训练 LSTMRNN
if __name__ == '__main__':
    # 搭建 LSTMRNN 模型
    model = Seq2Seq(vocab_size, target_vocab_size, buckets, size=5)
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=3)
    sess.run(tf.global_variables_initializer())
    # matplotlib可视化
    plt.ion()  # 设置连续 plot
    plt.show()
    # 训练多次
    for i in range(100):
        losses, points = model.step(sess, input_data, target_data, target_weights)
        x = range(inSize)
        plt.clf()
        plt.plot(x, target_data, 'r', x, points, 'b--')  #
        plt.draw()
        plt.pause(0.3)  # 每 0.3 s 刷新一次
        # 打印 cost 结果
        if i % 20 == 0:
            saver.save(sess, "./file/seq2seq_test.ckpt", global_step=i)
            print(losses)