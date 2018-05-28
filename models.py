# -*- code:utf-8 -*-
import tensorflow as tf
from functools import reduce


class Model(object):
    """
    Bug triage Models
    """

    def __init__(
            self, model_type, config_model):
        """

        intit
        """
        self.model_type = 'self._' + model_type + '()'
        self.config = config_model

        self.l2_loss = tf.constant(0.0)
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self.learning_rate = self.config['learning_rate']

        with tf.name_scope('Embedding'):
            self._embedding()

        with tf.name_scope('Convolution'):
            eval(self.model_type)

        with tf.name_scope('Cost'):
            self._cost()

        with tf.name_scope('Train'):
            self._train()

        with tf.name_scope('Evalution'):
            self._evaluation()

        with tf.name_scope('Summaries'):
            self._summary()

    def _inception_dense_net1_transpose(self):
        # 将通道位置放到最后一维，进行一维的inception式卷积
        with tf.name_scope('Inception'):
            # conv 1*100
            strides = [1, 1, 1, 1]
            embedding_shape = self.config['embedding_shape']
            # input channels
            num_outputs = embedding_shape[1]
            input_embedded = tf.transpose(self.input_embedded, [0, 1, 3, 2])
            with tf.name_scope('conv1'):
                pad_input = input_embedded
                # filter
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [1, 1, num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                a = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a += b_filter
            with tf.name_scope('conv3'):
                paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
                pad_input = tf.pad(input_embedded, paddings, "CONSTANT")
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [3, 1, num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                a3 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a3 += b_filter
            with tf.name_scope('conv5'):
                paddings = tf.constant([[0, 0], [2, 2], [0, 0], [0, 0]])
                pad_input = tf.pad(input_embedded, paddings, "CONSTANT")
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [5, 1, num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                a5 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a5 += b_filter
            with tf.name_scope('conv7'):
                paddings = tf.constant([[0, 0], [3, 3], [0, 0], [0, 0]])
                pad_input = tf.pad(input_embedded, paddings, "CONSTANT")
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [7, 1, num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                a7 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a7 += b_filter
            # max pooling
            with tf.name_scope('max_pooling'):
                paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
                pad_input = tf.pad(input_embedded, paddings, "CONSTANT")
                max_pooling = tf.nn.max_pool(
                    pad_input,
                    ksize=[1, 3, 1, 1],
                    strides=strides,
                    padding='VALID')
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [1, 1, num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                max_pooling = tf.nn.conv2d(
                    max_pooling,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                max_pooling += b_filter
            with tf.name_scope('concat'):
                outputs = tf.concat(
                    [a, a3, a5, a7, max_pooling],
                    axis=-1)

        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="keep_prob")

        # 先拼接后最大池化
        with tf.name_scope('flatten'):
            outputs = tf.nn.max_pool(
                outputs,
                ksize=[1, outputs.shape[1].value, 1, 1],
                strides=strides,
                padding='VALID')
            num_outputs = outputs.shape[-1].value
            outputs = tf.reshape(
                outputs,
                [-1, num_outputs])
            # Add dropout
            self.h_drop = tf.nn.dropout(
                outputs, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_outputs, self.config['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config['num_classes']]), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def _dpcnn(self):
            # 使用Tong Zhang的Deep Pyramid卷积模型

            with tf.name_scope('Inception'):
                # conv 1*100
                strides = [1, 1, 1, 1]
                embedding_shape = self.config['embedding_shape']
                # input channels
                num_outputs = 1 if not self.config['embedding_type'] ==\
                    'multiple_channels' else 2
                with tf.name_scope('conv1'):
                    pad_input = self.input_embedded
                    # filter
                    num_in = num_outputs
                    num_filter = 2 ** 7
                    # filter size 1*1
                    filter_shape = [1, embedding_shape[1], num_in, num_filter]
                    # filter weights matrix
                    W_filter = self._weight_variable(filter_shape)
                    b_filter = self._bias_variable([num_filter])
                    conv1 = tf.nn.conv2d(
                        pad_input,
                        W_filter,
                        strides=strides,
                        padding='VALID')
                    a = tf.nn.relu(tf.nn.bias_add(conv1, b_filter))
                with tf.name_scope('conv3'):
                    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
                    pad_input = tf.pad(
                        self.input_embedded, paddings, "CONSTANT")
                    num_in = num_outputs
                    num_filter = 2 ** 7
                    # filter size 1*1
                    filter_shape = [3, embedding_shape[1], num_in, num_filter]
                    # filter weights matrix
                    W_filter = self._weight_variable(filter_shape)
                    b_filter = self._bias_variable([num_filter])
                    conv3 = tf.nn.conv2d(
                        pad_input,
                        W_filter,
                        strides=strides,
                        padding='VALID')
                    a3 = tf.nn.relu(tf.nn.bias_add(conv3, b_filter))
                with tf.name_scope('conv5'):
                    paddings = tf.constant([[0, 0], [2, 2], [0, 0], [0, 0]])
                    pad_input = tf.pad(
                        self.input_embedded, paddings, "CONSTANT")
                    num_in = num_outputs
                    num_filter = 2 ** 7
                    # filter size 1*1
                    filter_shape = [5, embedding_shape[1], num_in, num_filter]
                    # filter weights matrix
                    W_filter = self._weight_variable(filter_shape)
                    b_filter = self._bias_variable([num_filter])
                    conv5 = tf.nn.conv2d(
                        pad_input,
                        W_filter,
                        strides=strides,
                        padding='VALID')
                    a5 = tf.nn.relu(tf.nn.bias_add(conv5, b_filter))
                with tf.name_scope('conv7'):
                    paddings = tf.constant([[0, 0], [3, 3], [0, 0], [0, 0]])
                    pad_input = tf.pad(
                        self.input_embedded, paddings, "CONSTANT")
                    num_in = num_outputs
                    num_filter = 2 ** 7
                    # filter size 1*1
                    filter_shape = [7, embedding_shape[1], num_in, num_filter]
                    # filter weights matrix
                    W_filter = self._weight_variable(filter_shape)
                    b_filter = self._bias_variable([num_filter])
                    conv7 = tf.nn.conv2d(
                        pad_input,
                        W_filter,
                        strides=strides,
                        padding='VALID')
                    a7 = tf.nn.relu(tf.nn.bias_add(conv7, b_filter))
                # max pooling
                with tf.name_scope('max_pooling'):
                    paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
                    pad_input = tf.pad(
                        self.input_embedded, paddings, "CONSTANT")
                    pad_input = tf.transpose(pad_input, [0, 1, 3, 2])
                    max_pooling = tf.nn.max_pool(
                        pad_input,
                        ksize=[1, 3, 1, 1],
                        strides=strides,
                        padding='VALID')
                    num_in = embedding_shape[1]
                    num_filter = 2 ** 7
                    # filter size 1*1
                    filter_shape = [1, 1, num_in, num_filter]
                    # filter weights matrix
                    W_filter = self._weight_variable(filter_shape)
                    b_filter = self._bias_variable([num_filter])
                    max_pooling = tf.nn.conv2d(
                        max_pooling,
                        W_filter,
                        strides=strides,
                        padding='VALID')
                    max_pooling = tf.nn.relu(
                        tf.nn.bias_add(max_pooling, b_filter))

                with tf.name_scope('concat'):
                    outputs = tf.concat(
                        [a, a3, a5, a7, max_pooling],
                        axis=-1)
                    # short_path = outputs
            # 拼接多个inception block
            # outputs = self._inception(outputs, name='incepiton1')
            # outputs += short_path

            with tf.name_scope('flatten'):
                outputs = tf.nn.max_pool(
                    outputs,
                    ksize=[1, outputs.shape[1].value, 1, 1],
                    strides=strides,
                    padding='VALID')
                num_outputs = outputs.shape[-1].value
                outputs = tf.reshape(
                    outputs,
                    [-1, num_outputs])
            # Add dropout
            with tf.name_scope("dropout"):
                self.dropout_keep_prob = tf.placeholder(
                    tf.float32, name="keep_prob")
                self.h_drop = tf.nn.dropout(
                    outputs, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[num_outputs, self.config['num_classes']],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(
                    0.1, shape=[self.config['num_classes']]), name="b")
                self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def _inception_dense_net(self):
        # 使用Dense nete 模型
        with tf.name_scope('Inception'):
            # conv 1*100
            strides = [1, 1, 1, 1]
            embedding_shape = self.config['embedding_shape']
            # input channels
            num_outputs = 1 if not self.config['embedding_type'] ==\
                'multiple_channels' else 2
            with tf.name_scope('conv1'):
                pad_input = self.input_embedded
                # filter
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [1, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                a = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a += b_filter
            with tf.name_scope('conv3'):
                paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
                pad_input = tf.pad(self.input_embedded, paddings, "CONSTANT")
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [3, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                a3 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a3 += b_filter
            with tf.name_scope('conv5'):
                paddings = tf.constant([[0, 0], [2, 2], [0, 0], [0, 0]])
                pad_input = tf.pad(self.input_embedded, paddings, "CONSTANT")
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [5, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                a5 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a5 += b_filter
            with tf.name_scope('conv7'):
                paddings = tf.constant([[0, 0], [3, 3], [0, 0], [0, 0]])
                pad_input = tf.pad(self.input_embedded, paddings, "CONSTANT")
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [7, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                a7 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a7 += b_filter
            # max pooling
            with tf.name_scope('max_pooling'):
                paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
                pad_input = tf.pad(self.input_embedded, paddings, "CONSTANT")
                pad_input = tf.transpose(pad_input, [0, 1, 3, 2])
                max_pooling = tf.nn.max_pool(
                    pad_input,
                    ksize=[1, 3, 1, 1],
                    strides=strides,
                    padding='VALID')
                num_in = embedding_shape[1]
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [1, 1, num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                max_pooling = tf.nn.conv2d(
                    max_pooling,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                max_pooling += b_filter
            with tf.name_scope('concat'):
                outputs = tf.concat(
                    [a, a3, a5, a7, max_pooling],
                    axis=-1)

        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="keep_prob")

        with tf.name_scope('inception_block'):
            current = outputs
            layers = 4
            growth = 4
            num_block = 1

            for block in range(num_block):
                with tf.name_scope('block' + str(block)):
                    if block:
                        current = self._transition(current)
                    current = self._block_inception(
                        current, layers, growth,
                        self.is_training, self.dropout_keep_prob)

        with tf.name_scope('flatten'):
            outputs = tf.nn.max_pool(
                current,
                ksize=[1, current.shape[1].value, 1, 1],
                strides=strides,
                padding='VALID')
            num_outputs = outputs.shape[-1].value
            outputs = tf.reshape(
                outputs,
                [-1, num_outputs])
            # Add dropout
            self.h_drop = tf.nn.dropout(
                outputs, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_outputs, self.config['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config['num_classes']]), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def _inception_dense_net1(self):
        # 仅使用 inception block结构
        with tf.name_scope('Inception'):
            # conv 1*100
            strides = [1, 1, 1, 1]
            embedding_shape = self.config['embedding_shape']
            # input channels
            num_outputs = 1 if not self.config['embedding_type'] ==\
                'multiple_channels' else 2
            with tf.name_scope('conv1'):
                pad_input = self.input_embedded
                # filter
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [1, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                a = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a += b_filter
            with tf.name_scope('conv3'):
                paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
                pad_input = tf.pad(self.input_embedded, paddings, "CONSTANT")
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [3, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                a3 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a3 += b_filter
            with tf.name_scope('conv5'):
                paddings = tf.constant([[0, 0], [2, 2], [0, 0], [0, 0]])
                pad_input = tf.pad(self.input_embedded, paddings, "CONSTANT")
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [5, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                a5 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a5 += b_filter
            with tf.name_scope('conv7'):
                paddings = tf.constant([[0, 0], [3, 3], [0, 0], [0, 0]])
                pad_input = tf.pad(self.input_embedded, paddings, "CONSTANT")
                num_in = num_outputs
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [7, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                a7 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a7 += b_filter
            # max pooling
            with tf.name_scope('max_pooling'):
                paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
                pad_input = tf.pad(self.input_embedded, paddings, "CONSTANT")
                pad_input = tf.transpose(pad_input, [0, 1, 3, 2])
                max_pooling = tf.nn.max_pool(
                    pad_input,
                    ksize=[1, 3, 1, 1],
                    strides=strides,
                    padding='VALID')
                num_in = embedding_shape[1]
                num_filter = 2 ** 7
                # filter size 1*1
                filter_shape = [1, 1, num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                max_pooling = tf.nn.conv2d(
                    max_pooling,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                max_pooling += b_filter
            with tf.name_scope('concat'):
                outputs = tf.concat(
                    [a, a3, a5, a7, max_pooling],
                    axis=-1)

        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="keep_prob")

        # with tf.name_scope('inception_block'):
        #     current = outputs
        #     layers = 1
        #     growth = 2 ** 7
        #     num_block = 1

        #     for block in range(num_block):
        #         current = self._block_inception(
        #             current, layers, growth,
        #             self.is_training, self.dropout_keep_prob,
        #             'block_inception' + str(block))

        with tf.name_scope('flatten'):
            outputs = tf.nn.max_pool(
                outputs,
                ksize=[1, outputs.shape[1].value, 1, 1],
                strides=strides,
                padding='VALID')
            num_outputs = outputs.shape[-1].value
            outputs = tf.reshape(
                outputs,
                [-1, num_outputs])
            # Add dropout
            self.h_drop = tf.nn.dropout(
                outputs, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_outputs, self.config['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config['num_classes']]), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def _text_inception_dense(self):
        # input channels
        num_in = 1 if not self.config['embedding_type'] ==\
            'multiple_channels' else 2

        # conv 1*1
        with tf.name_scope('Inception'):
            strides = [1, 1, 1, 1]
            embedding_shape = self.config['embedding_shape']
            with tf.name_scope('conv1'):
                # filter
                num_in = 1
                num_filter = 2 ** 4
                # filter size 1*1
                filter_shape = [1, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                conv1 = tf.nn.conv2d(
                    self.input_embedded,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a = tf.nn.relu(tf.nn.bias_add(conv1, b_filter))
            with tf.name_scope('conv3'):
                num_filter = 2 ** 8
                # filter size 1*1
                filter_shape = [3, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                conv3 = tf.nn.conv2d(
                    self.input_embedded,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a3 = tf.nn.relu(tf.nn.bias_add(conv3, b_filter))
            with tf.name_scope('conv5'):
                num_filter = 2 ** 8
                # filter size 1*1
                filter_shape = [5, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                conv5 = tf.nn.conv2d(
                    self.input_embedded,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a5 = tf.nn.relu(tf.nn.bias_add(conv5, b_filter))
            with tf.name_scope('conv7'):
                num_filter = 2 ** 4
                # filter size 1*1
                filter_shape = [7, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                conv7 = tf.nn.conv2d(
                    self.input_embedded,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a7 = tf.nn.relu(tf.nn.bias_add(conv7, b_filter))
            # max pooling
            with tf.name_scope('max_pooling'):
                num = self.input_embedded.shape[1].value
                a_max_pooling = tf.nn.max_pool(
                    self.input_embedded,
                    ksize=[1, num, 1, 1],
                    strides=strides,
                    padding='VALID')

            with tf.name_scope('concat'):
                a = tf.nn.max_pool(
                    a,
                    ksize=[1, 7, 1, 1],
                    strides=strides,
                    padding='VALID')
                a3 = tf.nn.max_pool(
                    a3,
                    ksize=[1, 5, 1, 1],
                    strides=strides,
                    padding='VALID')
                a5 = tf.nn.max_pool(
                    a5,
                    ksize=[1, 3, 1, 1],
                    strides=strides,
                    padding='VALID')
                a_concat = tf.concat([a, a3, a5, a7], axis=-1)
                tmp = a_concat.shape[1].value
                max_conv = tf.nn.max_pool(
                    a_concat,
                    ksize=[1, tmp, 1, 1],
                    strides=strides,
                    padding='VALID')
                outputs = tf.concat(
                    [tf.squeeze(max_conv, axis=[1, 2]),
                     tf.squeeze(a_max_pooling, axis=[1, 3])],
                    axis=-1)
                num_outputs = outputs.shape[-1].value

        # Add dropout
        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="keep_prob")
            h_drop = tf.nn.dropout(
                outputs, self.dropout_keep_prob)

        with tf.name_scope('pooling_block'):
            features = num_outputs
            current = h_drop
            layers = 4
            growth = 12
            num_block = 1

            for block in range(num_block):
                current, features = self._block_pooling(
                    current, layers, features, growth,
                    self.is_training, self.dropout_keep_prob,
                    'block_pooling' + str(block))

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[features, self.config['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config['num_classes']]), name="b")
            self.logits = tf.nn.xw_plus_b(current, W, b, name="scores")

    def _inception(self, outputs, name,
                   num_filter=2**7,
                   activate_function=tf.nn.relu):
        # 辅助函数 Inception block函数
        with tf.name_scope(name):
            # conv 1*100
            strides = [1, 1, 1, 1]
            num_outputs = outputs.shape[-1].value
            with tf.name_scope('conv1'):
                pad_input = outputs
                # filter
                num_in = num_outputs
                # filter size 1*1
                filter_shape = [1, 1, num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                conv1 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                if activate_function:
                    a = tf.nn.relu(tf.nn.bias_add(conv1, b_filter))
                else:
                    a = conv1 + b_filter
            with tf.name_scope('conv3'):
                paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
                pad_input = tf.pad(outputs, paddings, "CONSTANT")
                num_in = num_outputs
                # filter size 1*1
                filter_shape = [3, 1, num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                conv3 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                if activate_function:
                    a3 = tf.nn.relu(tf.nn.bias_add(conv3, b_filter))
                else:
                    a3 = conv3 + b_filter
            with tf.name_scope('conv5'):
                paddings = tf.constant([[0, 0], [2, 2], [0, 0], [0, 0]])
                pad_input = tf.pad(outputs, paddings, "CONSTANT")
                num_in = num_outputs
                # filter size 1*1
                filter_shape = [5, 1, num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                conv5 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                if activate_function:
                    a5 = tf.nn.relu(tf.nn.bias_add(conv5, b_filter))
                else:
                    a5 = conv5 + b_filter
            with tf.name_scope('conv7'):
                paddings = tf.constant([[0, 0], [3, 3], [0, 0], [0, 0]])
                pad_input = tf.pad(outputs, paddings, "CONSTANT")
                num_in = num_outputs
                # filter size 1*1
                filter_shape = [7, 1, num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                conv7 = tf.nn.conv2d(
                    pad_input,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                if activate_function:
                    a7 = tf.nn.relu(tf.nn.bias_add(conv7, b_filter))
                else:
                    a7 = conv7 + b_filter
            # max pooling
            with tf.name_scope('max_pooling'):
                paddings = tf.constant([[0, 0], [1, 1], [0, 0], [0, 0]])
                pad_input = tf.pad(outputs, paddings, "CONSTANT")
                max_pooling = tf.nn.max_pool(
                    pad_input,
                    ksize=[1, 3, 1, 1],
                    strides=strides,
                    padding='VALID')
                num_in = num_outputs
                # filter size 1*1
                filter_shape = [1, 1, num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                max_pooling = tf.nn.conv2d(
                    max_pooling,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                if activate_function:
                    max_pooling = tf.nn.relu(
                        tf.nn.bias_add(max_pooling, b_filter))
                else:
                    max_pooling += b_filter

            with tf.name_scope('concat'):
                outputs = tf.concat(
                    [a, a3, a5, a7, max_pooling],
                    axis=-1)
            return outputs

    def _text_inception(self):
        # inception module的尝试版
        # input channels
        num_in = 1 if not self.config['embedding_type'] ==\
            'multiple_channels' else 2

        # conv 1*1
        with tf.name_scope('Inception'):
            strides = [1, 1, 1, 1]
            embedding_shape = self.config['embedding_shape']
            with tf.name_scope('conv1'):
                # filter
                num_in = 1
                num_filter = 2 ** 4
                # filter size 1*1
                filter_shape = [1, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                conv1 = tf.nn.conv2d(
                    self.input_embedded,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a = tf.nn.relu(tf.nn.bias_add(conv1, b_filter))
            with tf.name_scope('conv3'):
                num_filter = 2 ** 8
                # filter size 1*1
                filter_shape = [3, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                conv3 = tf.nn.conv2d(
                    self.input_embedded,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a3 = tf.nn.relu(tf.nn.bias_add(conv3, b_filter))
            with tf.name_scope('conv5'):
                num_filter = 2 ** 8
                # filter size 1*1
                filter_shape = [5, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                conv5 = tf.nn.conv2d(
                    self.input_embedded,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a5 = tf.nn.relu(tf.nn.bias_add(conv5, b_filter))
            with tf.name_scope('conv7'):
                num_filter = 2 ** 4
                # filter size 1*1
                filter_shape = [7, embedding_shape[1], num_in, num_filter]
                # filter weights matrix
                W_filter = self._weight_variable(filter_shape)
                b_filter = self._bias_variable([num_filter])
                conv7 = tf.nn.conv2d(
                    self.input_embedded,
                    W_filter,
                    strides=strides,
                    padding='VALID')
                a7 = tf.nn.relu(tf.nn.bias_add(conv7, b_filter))
            # max pooling
            with tf.name_scope('max_pooling'):
                num = self.input_embedded.shape[1].value
                a_max_pooling = tf.nn.max_pool(
                    self.input_embedded,
                    ksize=[1, num, 1, 1],
                    strides=strides,
                    padding='VALID')

            with tf.name_scope('concat'):
                a = tf.nn.max_pool(
                    a,
                    ksize=[1, 7, 1, 1],
                    strides=strides,
                    padding='VALID')
                a3 = tf.nn.max_pool(
                    a3,
                    ksize=[1, 5, 1, 1],
                    strides=strides,
                    padding='VALID')
                a5 = tf.nn.max_pool(
                    a5,
                    ksize=[1, 3, 1, 1],
                    strides=strides,
                    padding='VALID')
                a_concat = tf.concat([a, a3, a5, a7], axis=-1)
                tmp = a_concat.shape[1].value
                max_conv = tf.nn.max_pool(
                    a_concat,
                    ksize=[1, tmp, 1, 1],
                    strides=strides,
                    padding='VALID')
                outputs = tf.concat(
                    [tf.squeeze(max_conv, axis=[1, 2]),
                     tf.squeeze(a_max_pooling, axis=[1, 3])],
                    axis=-1)
                num_outputs = outputs.shape[-1].value

        # Add dropout
        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="keep_prob")
            self.h_drop = tf.nn.dropout(
                outputs, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_outputs, self.config['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config['num_classes']]), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def _embedding(self):
        self.input_x = tf.placeholder(
            dtype=tf.int32,
            shape=[None, self.config['max_sent_length']],
            name='input_x')
        with tf.device('/cpu:0'):
            if not self.config['embedding_type'] == 'multiple_channels':
                if self.config['embedding_type'] == 'static':
                    self.embedding = tf.get_variable(
                        "embedding",
                        shape=self.config['embedding_shape'],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        dtype=tf.float32,
                        trainable=False)
                elif self.config['embedding_type'] == 'rand':
                    self.embedding = tf.get_variable(
                        "embedding",
                        shape=self.config['embedding_shape'],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        dtype=tf.float32,
                        trainable=True)
                elif self.config['embedding_type'] == 'non_static':
                    self.embedding = tf.get_variable(
                        "embedding",
                        shape=self.config['embedding_shape'],
                        initializer=tf.contrib.layers.xavier_initializer(),
                        dtype=tf.float32,
                        trainable=True)
                embedded = tf.nn.embedding_lookup(self.embedding, self.input_x)
                self.input_embedded = tf.expand_dims(embedded, -1)
            else:
                self.embedding = tf.get_variable(
                    "embedding",
                    shape=self.config['embedding_shape'],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    dtype=tf.float32,
                    trainable=False)
                self.another_embedding = tf.get_variable(
                    "another_embedding",
                    shape=self.config['embedding_shape'],
                    initializer=tf.contrib.layers.xavier_initializer(),
                    dtype=tf.float32,
                    trainable=True)
                embedded = tf.nn.embedding_lookup(self.embedding, self.input_x)
                expaned_embedded = tf.expand_dims(embedded, -1)
                another_embedded = tf.nn.embedding_lookup(
                    self.another_embedding, self.input_x)
                another_expaned_embedded = tf.expand_dims(another_embedded, -1)
                self.input_embedded = tf.concat(
                    [expaned_embedded, another_expaned_embedded], 3)

    def _text_dense(self):
        # block_pooling = bn+relu+fc

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        num_in = 1 if not self.config['embedding_type'] ==\
            'multiple_channels' else 2
        # normed_input = self._batch_norm_layer(
        #     self.input_embedded, self.is_training, 'embedded_input')
        for i, filter_size in enumerate(self.config['filter_sizes']):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size,
                                self.config['embedding_shape'][1],
                                num_in, self.config['num_filters']]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[self.config['num_filters']]), name="b")
                conv = tf.nn.conv2d(
                    self.input_embedded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config['max_sent_length'] -
                           filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        h = tf.concat(pooled_outputs, axis=-1)
        num_filters_total = h.get_shape().as_list()[-1]
        current = tf.reshape(h, [-1, num_filters_total])
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")
        current = tf.nn.dropout(current, self.dropout_keep_prob)
        with tf.name_scope('pooling_block'):
            features = num_filters_total
            layers = 4
            growth = 12
            num_block = 2

            for block in range(num_block):
                current, features = self._block_pooling(
                    current, layers, features, growth,
                    self.is_training, self.dropout_keep_prob,
                    'block_pooling' + str(block))

        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[features, self.config['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config['num_classes']]), name="b")
            self.logits = tf.nn.xw_plus_b(
                current, W, b, name="scores")

    def _text_conv_dense(self):
        """
        conv_dense = bn + relu + conv2d
        classifical dense nets
        """
        with tf.name_scope('conv1'):
            num_in = 1 if not self.config['embedding_type'] ==\
                'multiple_channels' else 2
            with tf.name_scope('conv'):
                W = self._weight_variable(
                    shape=[1, self.config['embedding_shape'][1],
                           num_in, self.config['num_filters'] * 3])
                conv1 = tf.nn.conv2d(
                    self.input_embedded, W, [1, 1, 1, 1], padding='VALID')
        with tf.name_scope('block_conv'):
            current = conv1

            features = self.config['num_filters'] * 3
            layers = 1
            growth = 12
            num_block = 1
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="dropout_keep_prob")
            for block in range(num_block - 1):
                current, features = self._block_conv(
                    current, layers, features, growth,
                    self.is_training, self.dropout_keep_prob,
                    'block_conv' + str(block))
                current = self._batch_activ_conv(
                    current, features,
                    features, 1,
                    self.is_training, self.dropout_keep_prob,
                    'block_conv_transition' + str(block))
                current = self._avg_pool(current, 2)
            current, features = self._block_conv(
                current, layers, features, growth,
                self.is_training, self.dropout_keep_prob,
                'block_conv_end')
            current = self._batch_activ_conv(
                current, features,
                features, 1, self.is_training,
                self.dropout_keep_prob,
                'block_conv_transition_end')
            final_dim = features * current.get_shape().as_list()[1]
            current = self._avg_pool(current, current.get_shape().as_list()[1])
        with tf.name_scope("output"):
            final_dim = features
            current = tf.reshape(current, [-1, final_dim])
            W = tf.get_variable(
                "W",
                shape=[final_dim, self.config['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config['num_classes']]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(current, W, b, name="scores")

    def _text_cnn_lstm(self):
        """


        经典CNN+LSTM
        """
        filter_sizes = self.config['filter_sizes']
        embedding_size = self.config['embedding_shape'][1]
        num_filters = self.config['num_filters']
        num_classes = self.config['num_classes']
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-L%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.input_embedded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                pooled_outputs.append(conv)
        num_features = pooled_outputs[-1].get_shape().as_list()[1]
        num_channels = len(pooled_outputs)
        with tf.name_scope("LSTM"):
            input_x = [tf.squeeze(x, 2) for x in pooled_outputs]
            input_x = reduce(lambda x, y: tf.concat(
                [x[:, :num_features, :],
                 y[:, :num_features, :]], axis=-1), input_x)
            input_x = tf.unstack(input_x, axis=2)
            lstm = tf.contrib.rnn.BasicLSTMCell(num_filters * num_channels)
            outputs, states = tf.contrib.rnn.static_rnn(
                lstm, input_x, dtype=tf.float32)
            self.lstm_out = outputs[-1]
        # Add dropout
        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="dropout_keep_prob")
            self.h_drop = tf.nn.dropout(
                self.lstm_out, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters * num_channels, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def _text_bilstm(self):
        """

        经典BiLSTM
        """
        n_hidden = self.config['n_hidden']
        num_classes = self.config['num_classes']
        batch_size = self.config['batch_size']
        with tf.name_scope("LSTM"):
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="dropout_keep_prob")
            input_x = tf.squeeze(self.input_embedded, -1)
            lstm_fw = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            lstm_bw = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            lstm_fw_init = lstm_fw.zero_state(batch_size, tf.float32)
            lstm_bw_init = lstm_bw.zero_state(batch_size, tf.float32)
            lstm_fw_drop = tf.contrib.rnn.DropoutWrapper(
                lstm_fw, output_keep_prob=self.dropout_keep_prob)
            lstm_bw_drop = tf.contrib.rnn.DropoutWrapper(
                lstm_bw, output_keep_prob=self.dropout_keep_prob)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_fw_drop, lstm_bw_drop,
                input_x, dtype=tf.float32,
                initial_state_fw=lstm_fw_init,
                initial_state_bw=lstm_bw_init)
            outputs = tf.concat(outputs, axis=-1)
            # lstm_output = self._attention(outputs)
            lstm_output = tf.squeeze(outputs[:, -1, :])
            # lstm_max = tf.squeeze(tf.reduce_max(outputs, axis=-2))
            # lstm_output = tf.concat([lstm_last, lstm_max], axis=-1)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            n_in = lstm_output.shape[-1].value
            W = tf.get_variable(
                "W",
                shape=[n_in, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.logits = tf.nn.xw_plus_b(lstm_output, W, b, name="scores")

    def _textlstm(self):
        """


        经典BasicLSTM
        """
        n_hidden = self.config['n_hidden']
        num_classes = self.config['num_classes']
        batch_size = self.config['batch_size']
        with tf.name_scope("LSTM"):
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="dropout_keep_prob")
            input_x = tf.squeeze(self.input_embedded, -1)
            lstm = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            lstm_init = lstm.zero_state(batch_size, tf.float32)
            lstm_drop = tf.contrib.rnn.DropoutWrapper(
                lstm, output_keep_prob=self.dropout_keep_prob)
            outputs, _ = tf.nn.dynamic_rnn(
                lstm_drop, input_x, dtype=tf.float32,
                initial_state=lstm_init)
            # lstm_output = tf.squeeze(outputs[:, -1, :])
            lstm_output = tf.squeeze(tf.reduce_max(outputs, axis=-1))
            # lstm_output = tf.concat([lstm_last, lstm_max], axis=-1)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            n_in = lstm_output.shape[-1].value
            W = tf.get_variable(
                "W",
                shape=[n_in, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.logits = tf.nn.xw_plus_b(lstm_output, W, b, name="scores")

    def _text_dp_cnn(self):
        # 多层CNN结构
        with tf.name_scope('conv1'):
            num_in = 1 if not self.config['embedding_type'] ==\
                'multiple_channels' else 2
            with tf.name_scope('conv'):
                W = self._weight_variable(
                    shape=[1, self.config['embedding_shape'][1],
                           num_in, self.config['num_filters'] * 3])
                conv1 = tf.nn.conv2d(
                    self.input_embedded, W, [1, 1, 1, 1], padding='VALID')
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")
        with tf.name_scope('dp'):
            features = self.config['num_filters'] * 3
            current = conv1
            layers = 2
            blocks = 5
            for block in range(blocks):
                current = self._block_dp(
                    current, layers,
                    features, features, self.is_training,
                    self.dropout_keep_prob, str(block))
                current = self._avg_pool(current, 2)

        final_dim = features * current.get_shape().as_list()[1]
        current = self._avg_pool(current, current.get_shape().as_list()[1])
        with tf.name_scope("output"):
            final_dim = features
            current = tf.reshape(current, [-1, final_dim])
            W = tf.get_variable(
                "W",
                shape=[final_dim, self.config['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config['num_classes']]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(current, W, b, name="scores")

    def _hierarchical_cnn(self):
        """


        多层cnn结构，每层加pool层进行输出
        """
        with tf.name_scope("m-conv"):

            filter_size = self.config['filter_sizes'][0]
            num_filters = self.config['num_filters']
            embedding_size = self.config['embedding_shape'][1]
            num_classes = self.config['num_classes']
            # Convolution Layer
            filter_shape = [filter_size, embedding_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(
                filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(
                0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                self.input_embedded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pool1 = tf.nn.max_pool(
                h,
                ksize=[1, h.get_shape().as_list()[1], 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            filter_shape = [filter_size, 1, num_filters, num_filters]
            W = tf.Variable(tf.truncated_normal(
                filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(
                0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                conv,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            pool2 = tf.nn.max_pool(
                h,
                ksize=[1, h.get_shape().as_list()[1], 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            filter_shape = [filter_size, 1, num_filters, num_filters]
            W = tf.Variable(tf.truncated_normal(
                filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(
                0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                conv,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pool3 = tf.nn.max_pool(
                h,
                ksize=[1, h.get_shape().as_list()[1], 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs = [pool1, pool2, pool3]
            pool = tf.concat(pooled_outputs, 3)
            self.h_pool = tf.reshape(pool, [-1, num_filters * 3])

        # Add dropout
        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="dropout_keep_prob")
            self.h_drop = tf.nn.dropout(
                self.h_pool, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters * 3, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def _multi_layers_cnn(self):
        """

        三层CNN叠加模型
        """
        filter_size = self.config['filter_sizes'][-1]
        filter_shape = [filter_size, self.config['embedding_shape']
                        [1], 1, self.config['num_filters'] * 3]
        W = tf.Variable(tf.truncated_normal(
            filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(
            0.1, shape=[self.config['num_filters'] * 3]), name="b")
        conv = tf.nn.conv2d(
            self.input_embedded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        filter_shape = [filter_size, 1,
                        self.config['num_filters'] * 3,
                        self.config['num_filters'] * 6]
        W = tf.Variable(tf.truncated_normal(
            filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(
            0.1, shape=[self.config['num_filters'] * 6]), name="b")
        conv = tf.nn.conv2d(
            conv,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        filter_shape = [filter_size, 1,
                        self.config['num_filters'] * 6,
                        self.config['num_filters'] * 9]
        W = tf.Variable(tf.truncated_normal(
            filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(
            0.1, shape=[self.config['num_filters'] * 9]), name="b")
        conv = tf.nn.conv2d(
            conv,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        filter_shape = [h.shape[1].value, 1,
                        self.config['num_filters'] * 9,
                        self.config['num_filters'] * 9]
        W = tf.Variable(tf.truncated_normal(
            filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(
            0.1, shape=[self.config['num_filters'] * 9]), name="b")
        conv = tf.nn.conv2d(
            conv,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # # Maxpooling over the outputs
        # pooled = tf.nn.max_pool(
        #     h,
        #     ksize=[1, h.get_shape().as_list()[1], 1, 1],
        #     strides=[1, 1, 1, 1],
        #     padding='VALID',
        #     name="pool")
        # self.h_pool = tf.reshape(
        # pooled, [-1, self.config['num_filters'] * 9])
        h = tf.squeeze(h)
        # Add dropout
        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="dropout_keep_prob")
            self.h_drop = tf.nn.dropout(
                h, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[self.config['num_filters'] * 9,
                       self.config['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config['num_classes']]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def _textcnn(self):
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        num_in = 1 if not self.config['embedding_type'] ==\
            'multiple_channels' else 2
        # normed_input = self._batch_norm_layer(
        #     self.input_embedded, self.is_training, 'bn_embedding')
        for i, filter_size in enumerate(self.config['filter_sizes']):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size,
                                self.config['embedding_shape'][1],
                                num_in, self.config['num_filters']]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[self.config['num_filters']]), name="b")
                conv = tf.nn.conv2d(
                    self.input_embedded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config['max_sent_length'] -
                           filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.config['num_filters'] *\
            len(self.config['filter_sizes'])
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="dropout_keep_prob")
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.config['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config['num_classes']]), name="b")
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def _down_sampling(self, outputs, name):
        # 降采样函数
        with tf.name_scope(name):
            outputs = tf.nn.max_pool(
                outputs,
                ksize=[1, 3, 1, 1],
                strides=[1, 2, 1, 1],
                padding='VALID')
        return outputs

    def _transition(self, current):
        with tf.name_scope('transition'):
            # filter
            num_in = num_filter = current.shape[-1].value
            # filter size 1*1
            filter_shape = [3, 1, num_in, num_filter]
            # filter weights matrix
            W_filter = self._weight_variable(filter_shape)
            b_filter = self._bias_variable([num_filter])
            conv1 = tf.nn.conv2d(
                current,
                W_filter,
                strides=[1, 2, 1, 1],
                padding='VALID')
            return tf.nn.relu(tf.nn.bias_add(conv1, b_filter))

    def _attention(self, hidden_state):
        """
        暂时还没有用
        Hierarchical Attention Network
        """
        n_hidden = hidden_state.shape[-1].value
        len_sent = hidden_state.shape[-2].value
        hidden_state_flat = tf.reshape(hidden_state, [-1, n_hidden])

        W = self._weight_variable([n_hidden, n_hidden])
        b = self._bias_variable([n_hidden])
        hidden_represention = tf.nn.tanh(tf.matmul(hidden_state_flat, W) + b)
        hidden_represention = tf.reshape(
            hidden_represention, [-1, len_sent, n_hidden])

        context_vector = self._weight_variable([n_hidden])
        hidden_state_context_similiarity = tf.multiply(
            hidden_represention, context_vector)
        attention_logits = tf.reduce_sum(
            hidden_state_context_similiarity, axis=-1)
        attention_logits_max = tf.reduce_max(
            attention_logits, axis=-1, keep_dims=True)
        p_attention = tf.nn.softmax(attention_logits - attention_logits_max)
        p_attention_expanded = tf.expand_dims(p_attention, axis=-1)

        sentence_representation = tf.multiply(
            p_attention_expanded, hidden_state)
        sentence_representation = tf.reduce_sum(
            sentence_representation, axis=-2)
        return sentence_representation

    def _block_inception(self, input, layers,
                         growth, is_training,
                         keep_prob):
        '''
        弃用
        '''
        current = input
        for idx in range(layers):
            with tf.name_scope("layer" + str(idx)):
                tmp = self._batch_activ_inception(
                    current, growth, is_training,
                    keep_prob)
                current = tf.concat([current, tmp], axis=-1)
        return current

    def _batch_activ_inception(self, current, out_features,
                               is_training, keep_prob):
        current = tf.nn.relu(current)
        current = self._inception(
            current, name='inception',
            num_filter=out_features, activate_function=None)
        current = tf.nn.dropout(current, keep_prob)
        return current

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def _conv2d(self, input, in_features,
                out_features, kernel_size,
                with_bias=False):
        W = self._weight_variable(
            [kernel_size, 1, in_features, out_features])
        conv = tf.nn.conv2d(input, W, [1, 1, 1, 1], padding='SAME')
        if with_bias:
            return conv + self._bias_variable([out_features])
        return conv

    def _fc(self, input, in_features, out_features):
        W = self._weight_variable([in_features, out_features])
        b = self._bias_variable([out_features])
        fc = tf.matmul(input, W) + b
        return fc

    def _batch_activ_fc(self, current, in_features,
                        out_features, is_training,
                        keep_prob, scope_bn):

        current = self._batch_norm_layer(
            current, is_training, scope_bn)
        current = tf.nn.relu(current)
        current = self._fc(current, in_features, out_features)
        # current = tf.nn.dropout(current, keep_prob)
        return current

    def _batch_activ_conv(self, current, in_features,
                          out_features, kernel_size,
                          is_training, keep_prob, scope_bn):
        current = self._batch_norm_layer(
            current, is_training, scope_bn)
        current = tf.nn.relu(current)
        current = self._conv2d(
            current, in_features, out_features, kernel_size)
        # current = tf.nn.dropout(current, keep_prob)
        return current

    def _avg_pool(self, input, s):
        return tf.nn.avg_pool(input, [1, s, 1, 1], [1, s, 1, 1], 'VALID')

    def _block_dp(self, input_,
                  layers, in_features,
                  out_features, is_training,
                  keep_prob, scope_bn):
        current = input_
        for idx in range(layers):
            current = self._conv2d(current, in_features, out_features, 3, True)
        current = tf.add(current, input_)
        return current

    def _block_conv(self, input, layers,
                    in_features, growth,
                    is_training, keep_prob,
                    scope_bn):
        current = input
        features = in_features
        for idx in range(layers):
            tmp = self._batch_activ_conv(
                current, features, growth,
                3, is_training, keep_prob, scope_bn)
            current = tf.concat((current, tmp), axis=-1)
            features += growth
        return current, features

    def _block_pooling(self, input, layers,
                       in_features, growth,
                       is_training, keep_prob,
                       scope_bp='block_pooling'):
        current = input
        features = in_features
        with tf.name_scope(scope_bp):
            for idx in range(layers):
                temp = self._batch_activ_fc(
                    current, features, growth,
                    is_training, keep_prob, scope_bp + str(idx))
                current = tf.concat([current, temp], axis=-1)
                features += growth
        return current, features

    def _batch_norm_layer(self, x, train_phase, scope_bn):
        with tf.variable_scope(scope_bn):
            beta = tf.Variable(tf.constant(
                0.0, shape=[x.shape[-1]]), name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(
                1.0, shape=[x.shape[-1]]), name='gamma', trainable=True)
            axises = list(range(len(x.shape) - 1))
            batch_mean, batch_var = tf.nn.moments(
                x, axises, name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            mean, var = tf.cond(
                train_phase,
                mean_var_with_update,
                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(
                x, mean, var, beta, gamma, 1e-3)
        return normed

    def _cost(self):
        self.input_y = tf.placeholder(
            tf.int64, [None, self.config['num_classes']], name="input_y")
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.input_y)
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(v) for v in tf.trainable_variables()
             if 'bias' not in v.name])
        self.cost = tf.reduce_mean(losses) +\
            self.config['l2_reg_lambda'] * l2_loss

    def _train(self):
        self.global_step = tf.Variable(
            0, name="global_step", trainable=False)
        # lr = tf.train.exponential_decay(self.config['learning_rate'],
        #                                 self.global_step,
        #                                 self.config['decay_steps'],
        #                                 self.config['decay_rate'],
        #                                 staircase=True)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = optimizer.compute_gradients(self.cost)
        self.train_op = optimizer.apply_gradients(
            self.grads_and_vars, global_step=self.global_step)

    def _evaluation(self):
        with tf.name_scope('compute_accuray'):
            self.label = tf.argmax(self.input_y, 1)
            self.prediction = tf.argmax(self.logits, 1)

            # training accuracy
            correct_at_1 = tf.nn.in_top_k(self.logits, self.label, 1)
            self.accuracy_at_1 =\
                tf.reduce_mean(tf.cast(correct_at_1, tf.float32))

        with tf.name_scope("test_accuracy"):
            self.precision_at_1, self.precision_op_at_1 = \
                tf.contrib.metrics.streaming_sparse_precision_at_k(
                    predictions=self.logits,
                    labels=self.label,
                    k=1,
                    name="precision_at_1")
            self.recall_at_1, self.recall_op_at_1 = \
                tf.contrib.metrics.streaming_sparse_recall_at_k(
                    predictions=self.logits,
                    labels=self.label,
                    k=1,
                    name="recall_at1")
            self.precision_at_2, self.precision_op_at_2 = \
                tf.contrib.metrics.streaming_sparse_precision_at_k(
                    predictions=self.logits,
                    labels=self.label,
                    k=2,
                    name="precision_at_2")
            self.recall_at_2, self.recall_op_at_2 = \
                tf.contrib.metrics.streaming_sparse_recall_at_k(
                    predictions=self.logits,
                    labels=self.label,
                    k=2,
                    name="recall_at2")
            self.precision_at_3, self.precision_op_at_3 = \
                tf.contrib.metrics.streaming_sparse_precision_at_k(
                    predictions=self.logits,
                    labels=self.label,
                    k=3,
                    name="precision_at_3")
            self.recall_at_3, self.recall_op_at_3 = \
                tf.contrib.metrics.streaming_sparse_recall_at_k(
                    predictions=self.logits,
                    labels=self.label,
                    k=3,
                    name="recall_at3")
            self.precision_at_4, self.precision_op_at_4 = \
                tf.contrib.metrics.streaming_sparse_precision_at_k(
                    predictions=self.logits,
                    labels=self.label,
                    k=4,
                    name="precision_at_4")
            self.recall_at_4, self.recall_op_at_4 = \
                tf.contrib.metrics.streaming_sparse_recall_at_k(
                    predictions=self.logits,
                    labels=self.label,
                    k=4,
                    name="recall_at4")
            self.precision_at_5, self.precision_op_at_5 = \
                tf.contrib.metrics.streaming_sparse_precision_at_k(
                    predictions=self.logits,
                    labels=self.label,
                    k=5,
                    name="precision_at_5")
            self.recall_at_5, self.recall_op_at_5 = \
                tf.contrib.metrics.streaming_sparse_recall_at_k(
                    predictions=self.logits,
                    labels=self.label,
                    k=5,
                    name="recall_at5")
            self.recall_at_10, self.recall_op_at_10 = \
                tf.contrib.metrics.streaming_sparse_recall_at_k(
                    predictions=self.logits,
                    labels=self.label,
                    k=10,
                    name="recall_at_10")

        with tf.name_scope("prdiction_top_k"):
            lg = tf.nn.softmax(self.logits)
            self.prediction_top_k_values, self.prediction_top_k_indices = \
                tf.nn.top_k(lg, 15)

    def _summary(self):
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in self.grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name),
                    tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Summaries for loss and accuracy

        loss_summary = tf.summary.scalar("cost", self.cost)
        streaming_accuray_summary =\
            tf.summary.scalar("streaming_accuray",
                              self.recall_op_at_1)
        accuracy_summary_at_1 = \
            tf.summary.scalar("accuracy_at_1", self.accuracy_at_1)

        # Train Summaries
        self.train_summary_op = tf.summary.merge(
            [loss_summary,
             accuracy_summary_at_1,
             grad_summaries_merged])

        # test summaries
        self.test_summary_op = tf.summary.merge(
            [loss_summary,
             streaming_accuray_summary])

# if __name__ == '__main__':
#     mc = selfModel('')
