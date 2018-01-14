# -*- code:utf-8 -*-
import tensorflow as tf
import numpy as np
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
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def _multi_layers_cnn(self):
        """

        三层CNN叠加
        """
        filter_size = self.config['filter_sizes'][0]
        filter_shape = [filter_size, self.config['embedding_shape']
                        [1], 1, self.config['num_filters']]
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
        filter_shape = [filter_size, 1,
                        self.config['num_filters'], self.config['num_filters']]
        W = tf.Variable(tf.truncated_normal(
            filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(
            0.1, shape=[self.config['num_filters']]), name="b")
        conv = tf.nn.conv2d(
            conv,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        filter_shape = [filter_size, 1,
                        self.config['num_filters'], self.config['num_filters']]
        W = tf.Variable(tf.truncated_normal(
            filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(
            0.1, shape=[self.config['num_filters']]), name="b")
        conv = tf.nn.conv2d(
            conv,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        # Maxpooling over the outputs
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, h.get_shape().as_list()[1], 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        self.h_pool = tf.reshape(pooled, [-1, self.config['num_filters']])

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
                shape=[self.config['num_filters'], self.config['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config['num_classes']]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

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

    def _textlstm(self):
        """


        经典BasicLSTM
        """
        n_hidden = self.config['n_hidden']
        sequence_length = self.config['max_sent_length']
        num_classes = self.config['num_classes']
        with tf.name_scope("LSTM"):
            input_x = tf.squeeze(self.input_embedded, -1)
            input_x = tf.unstack(input_x, axis=1)
            # lstm = tf.contrib.rnn.LSTMCell(num_filters, use_peepholes=True)
            lstm = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
            outputs, _ = tf.contrib.rnn.static_rnn(
                lstm, input_x, dtype=tf.float32)
            # pdb.set_trace()
            # self.lstm_out = outputs[-1]
            self.lstm_out = tf.concat(outputs, axis=1)

        # Add dropout
        with tf.name_scope("dropout"):
            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="dropout_keep_prob")
            self.h_drop = tf.nn.dropout(self.lstm_out, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[n_hidden * sequence_length, num_classes],
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
        sequence_length = self.config['max_sent_length']
        num_classes = self.config['num_classes']
        with tf.name_scope("BiLSTM"):
            input_x = tf.squeeze(self.input_embedded, -1)
            input_x = tf.unstack(input_x, axis=1)
            lstm_forward = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            lstm_backward = tf.contrib.rnn.BasicLSTMCell(n_hidden)
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                lstm_forward, lstm_backward, input_x, dtype=tf.float32)
            self.lstm_out = tf.concat(outputs, axis=1)

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
                shape=[sequence_length * n_hidden * 2, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

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

    def _text_dense(self):
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
        num_filters_total = self.config['num_filters'] *\
            len(self.config['filter_sizes'])
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.name_scope('pooling_block'):
            current = self.h_pool_flat
            features = num_filters_total
            layers = 1
            growth = 3
            num_block = 1

            self.dropout_keep_prob = tf.placeholder(
                tf.float32, name="dropout_keep_prob")
            for block in range(num_block):
                features = num_filters_total
                current, features = self._block_pooling(
                    current, layers, features, growth,
                    self.is_training, self.dropout_keep_prob,
                    'block_pooling' + str(block))
                current = self._fc(current, features, num_filters_total)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.config['num_classes']],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.config['num_classes']]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(current, W, b, name="scores")

    def _text_conv_dense(self):
        with tf.name_scope('conv1'):
            normed_input = self._batch_norm_layer(
                self.input_embedded, self.is_training, 'conv1_bn')
            num_in = 1 if not self.config['embedding_type'] ==\
                'multiple_channels' else 2
            with tf.name_scope('conv1'):
                W = self._weight_variable(
                    shape=[3, self.config['embedding_shape'][1],
                           num_in, num_in])
                conv1 = tf.nn.conv2d(
                    normed_input, W, [1, 1, 1, 1], padding='VALID')
        with tf.name_scope('block_conv'):
            current = conv1
            features = num_in
            layers = 2
            growth = 3
            num_block = 2
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
            [kernel_size, kernel_size, in_features, out_features])
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
        current = tf.nn.dropout(current, keep_prob)
        return current

    def _batch_activ_conv(self, current, in_features,
                          out_features, kernel_size,
                          is_training, keep_prob, scope_bn):
        current = self._batch_norm_layer(
            current, is_training, scope_bn)
        current = tf.nn.relu(current)
        current = self._conv2d(
            current, in_features, out_features, kernel_size)
        current = tf.nn.dropout(current, keep_prob)
        return current

    def _avg_pool(self, input, s):
        return tf.nn.avg_pool(input, [1, s, 1, 1], [1, s, 1, 1], 'VALID')

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
            axises = np.arange(len(x.shape) - 1)
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
        self.cost = tf.reduce_mean(losses) +\
            self.config['l2_reg_lambda'] * self.l2_loss

    def _train(self):
        self.global_step = tf.Variable(
            0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.config['learning_rate'])
        self.grads_and_vars = optimizer.compute_gradients(self.cost)
        self.train_op = optimizer.apply_gradients(
            self.grads_and_vars, global_step=self.global_step)

    def _evaluation(self):
        with tf.name_scope('compute_accuray'):
            self.label = tf.arg_max(self.input_y, 1)
            self.prediction = tf.arg_max(self.logits, 1)

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
