# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 256  # 词向量维度
    seq_length = 1600  # 序列长度
    num_classes = 6  # 类别数 #10
    num_filters = [256,256,256]  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 8000  # 词汇表达小  #5000

    hidden_dim = 256  # 全连接层神经元

    dropout_keep_prob = 0.8  # dropout保留比例
    learning_rate = 5e-4  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 15  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    filter_sizes=[3,4,5]

class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/gpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        pooled_outputs=[]
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # CNN layer
                conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters[i], 
                                        filter_size, name='conv%s' % filter_size,padding='same',
                                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
                # global max pooling layer
                gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp%s' % filter_size)
                pooled_outputs.append(gmp)

        #num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs,1)
        #print(self.h_pool)
        #self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        #print(self.h_pool_flat)

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc2 = tf.layers.dense(self.h_pool, self.config.hidden_dim, name='fc1')
            fc2 = tf.contrib.layers.dropout(fc2, self.keep_prob)
            b = tf.Variable(tf.constant(0.1, shape=[self.config.hidden_dim]), name="b2")
            fc2 = tf.nn.relu(tf.nn.bias_add(fc2, b), name="relu2")

            fc = tf.layers.dense(fc2, self.config.hidden_dim, name='fc3')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            b = tf.Variable(tf.constant(0.1, shape=[self.config.hidden_dim]), name="b")
            fc = tf.nn.relu(tf.nn.bias_add(fc, b), name="relu")

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
