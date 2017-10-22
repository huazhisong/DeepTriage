# -*- coding:utf-8 -*-
import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed
    by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, num_filters, batch_size,
            filter_sizes=list(), top_k=3,
            embedding_type=None, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(
            tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(
            tf.int64, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(
            tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            if embedding_type == 'static':
                self.W = tf.Variable(
                    tf.random_uniform(
                        [vocab_size, embedding_size],
                        -1.0, 1.0),
                    trainable=False,
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(
                    self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(
                    self.embedded_chars, -1)
            elif embedding_type == 'rand' or embedding_type == 'none_static':
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                self.embedded_chars = tf.nn.embedding_lookup(
                    self.W, self.input_x)
                self.embedded_chars_expanded = tf.expand_dims(
                    self.embedded_chars, -1)
            elif embedding_type == 'multiple_channels':
                self.W = tf.Variable(
                    tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                    name="W")
                train_embedded_chars = tf.nn.embedding_lookup(
                    self.W, self.input_x)
                train_embedded_chars = tf.expand_dims(train_embedded_chars, -1)
                self.W_static = tf.Variable(
                    tf.random_uniform(
                        [vocab_size, embedding_size],
                        -1.0, 1.0),
                    trainable=False,
                    name="W_static")
                static_embedded_chars = tf.nn.embedding_lookup(
                    self.W_static, self.input_x)
                static_embedded_chars = tf.expand_dims(
                    static_embedded_chars, -1)
                self.embedded_chars_expanded = tf.concat(
                    [train_embedded_chars, static_embedded_chars], 3)
            else:
                print("\n**\nWrong embedding type!\n**\n")

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(
                self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        label = tf.arg_max(self.input_y, 1)
        prediction = tf.arg_max(self.logits, 1)
        with tf.name_scope("prdiction_top_k"):
            _, self.prediction_top_k_indice = \
                tf.nn.top_k(self.logits, 15)

        # Accuracy
        with tf.name_scope("accuracy"):
            self.correct_at_1 = tf.nn.in_top_k(self.logits, label, 1)
            self.accuracy_at_1 =\
                tf.reduce_mean(tf.cast(self.correct_at_1, tf.float32))
            self.correct_at_2 = tf.nn.in_top_k(self.logits, label, 2)
            self.accuracy_at_2 = \
                tf.reduce_mean(tf.cast(self.correct_at_2, tf.float32))
            self.correct_at_3 = tf.nn.in_top_k(self.logits, label, 3)
            self.accuracy_at_3 = \
                tf.reduce_mean(tf.cast(self.correct_at_3, tf.float32))
            self.correct_at_4 = tf.nn.in_top_k(self.logits, label, 4)
            self.accuracy_at_4 = \
                tf.reduce_mean(tf.cast(self.correct_at_4, tf.float32))
            self.correct_at_5 = tf.nn.in_top_k(self.logits, label, 5)
            self.accuracy_at_5 = \
                tf.reduce_mean(tf.cast(self.correct_at_5, tf.float32))
        # Evaluation
        with tf.name_scope("evaluation"):
            self.streaming_accuracy, self.streaming_accuray_op = \
                tf.contrib.metrics.streaming_accuracy(
                    predictions=prediction,
                    labels=label,
                    name='streaming_accuracy')
            self.precision_at_1, self.precision_op_at_1 = \
                tf.contrib.metrics.streaming_sparse_precision_at_k(
                    predictions=self.logits,
                    labels=label,
                    k=1,
                    name="precision_at_1")
            self.recall_at_1, self.recall_op_at_1 = \
                tf.contrib.metrics.streaming_sparse_recall_at_k(
                    predictions=self.logits,
                    labels=label,
                    k=1,
                    name="recall_at1")
            self.precision_at_2, self.precision_op_at_2 = \
                tf.contrib.metrics.streaming_sparse_precision_at_k(
                    predictions=self.logits,
                    labels=label,
                    k=2,
                    name="precision_at_1")
            self.recall_at_2, self.recall_op_at_2 = \
                tf.contrib.metrics.streaming_sparse_recall_at_k(
                    predictions=self.logits,
                    labels=label,
                    k=2,
                    name="recall_at1")
            self.precision_at_3, self.precision_op_at_3 = \
                tf.contrib.metrics.streaming_sparse_precision_at_k(
                    predictions=self.logits,
                    labels=label,
                    k=3,
                    name="precision_at_1")
            self.recall_at_3, self.recall_op_at_3 = \
                tf.contrib.metrics.streaming_sparse_recall_at_k(
                    predictions=self.logits,
                    labels=label,
                    k=3,
                    name="recall_at1")
            self.precision_at_4, self.precision_op_at_4 = \
                tf.contrib.metrics.streaming_sparse_precision_at_k(
                    predictions=self.logits,
                    labels=label,
                    k=4,
                    name="precision_at_1")
            self.recall_at_4, self.recall_op_at_4 = \
                tf.contrib.metrics.streaming_sparse_recall_at_k(
                    predictions=self.logits,
                    labels=label,
                    k=4,
                    name="recall_at1")
            self.precision_at_5, self.precision_op_at_5 = \
                tf.contrib.metrics.streaming_sparse_precision_at_k(
                    predictions=self.logits,
                    labels=label,
                    k=5,
                    name="precision_at_1")
            self.recall_at_5, self.recall_op_at_5 = \
                tf.contrib.metrics.streaming_sparse_recall_at_k(
                    predictions=self.logits,
                    labels=label,
                    k=5,
                    name="recall_at1")
