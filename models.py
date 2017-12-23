# -*- code:utf-8 -*-
import tensorflow as tf


class Model(object):
    """
    A self for text classification.
    Uses an embedding layer, followed
    by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, model_type,
            max_sent_length, num_classes,
            num_filters, filter_sizes,
            embedding_type, embedding_size,
            embedding, l2_reg_lambda,
            learning_rate):
        """


        intit
        """
        self.model_type = 'self._' + model_type + '()'
        self.num_classes = num_classes
        self.max_sent_length = max_sent_length
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.embedding_type = embedding_type
        self.embedding_size = embedding_size
        self.l2_reg_lambda = l2_reg_lambda
        self.learning_rate = learning_rate
        self.embedding = embedding

        self.l2_loss = tf.constant(0.0)
        with tf.name_scope('Embedding'):
            self.embedded_input = self._embedding()

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
            dtype=tf.int32, shape=[None, self.max_sent_length], name='input_x')
        with tf.device('/cpu:0'):
            if not self.embedding_type == 'multiple_channels':
                if self.embedding_type == 'static':
                    self.embedding = tf.get_variable(
                        "embedding",
                        shape=self.embedding.shape,
                        initializer=tf.constant_initializer(self.embedding),
                        dtype=tf.float32,
                        trainable=False)
                elif self.embedding_type == 'rand':
                    self.embedding = tf.get_variable(
                        "embedding",
                        shape=self.embedding.shape,
                        initializer=tf.contrib.layers.xavier_initializer(),
                        dtype=tf.float32,
                        trainable=True)
                elif self.embedding_type == 'non_static':
                    self.embedding = tf.get_variable(
                        "embedding",
                        shape=self.embedding.shape,
                        initializer=tf.constant_initializer(self.embedding),
                        dtype=tf.float32,
                        trainable=True)
                embedded = tf.nn.embedding_lookup(self.embedding, self.input_x)
                self.input_embedded = tf.expand_dims(embedded, -1)
            else:
                self.embedding = tf.get_variable(
                    "embedding",
                    shape=self.embedding.shape,
                    initializer=tf.constant_initializer(self.embedding),
                    dtype=tf.float32,
                    trainable=False)
                self.another_embedding = tf.get_variable(
                    "another_embedding",
                    shape=self.embedding.shape,
                    initializer=tf.constant_initializer(self.embedding),
                    dtype=tf.float32,
                    trainable=True)
                embedded = tf.nn.embedding_lookup(self.embedding, self.input_x)
                expaned_embedded = tf.expand_dims(embedded, -1)
                another_embedded = tf.nn.embedding_lookup(
                    self.another_embedding, self.input_x)
                another_expaned_embedded = tf.expand_dims(another_embedded, -1)
                self.input_embedded = tf.concat(
                    [expaned_embedded, another_expaned_embedded], 3)

    def _classical_model(self):
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        num_in = 1 if not self.embedding_type == 'multiple_channels' else 2
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size,
                                self.embedding_size,
                                num_in, self.num_filters]
                W = tf.Variable(tf.truncated_normal(
                    filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(
                    0.1, shape=[self.num_filters]), name="b")
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
                    ksize=[1, self.max_sent_length -
                           filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
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
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(
                0.1, shape=[self.num_classes]), name="b")
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")

    def _cost(self):
        self.input_y = tf.placeholder(
            tf.int64, [None, self.num_classes], name="input_y")
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.logits, labels=self.input_y)
        self.cost = tf.reduce_mean(losses) + self.l2_reg_lambda * self.l2_loss

    def _train(self):
        self.global_step = tf.Variable(
            0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.grads_and_vars = optimizer.compute_gradients(self.cost)
        self.train_op = optimizer.apply_gradients(
            self.grads_and_vars, global_step=self.global_step)

    def _evaluation(self):
        self.label = tf.arg_max(self.input_y, 1)
        self.prediction = tf.arg_max(self.logits, 1)
        self.streaming_accuracy, self.streaming_accuray_op = \
            tf.contrib.metrics.streaming_accuracy(
                predictions=self.prediction,
                labels=self.label,
                name='streaming_accuracy')

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
        with tf.name_scope('compute_accuray'):
            self.label = tf.arg_max(self.input_y, 1)
            self.prediction = tf.arg_max(self.logits, 1)
            self.streaming_accuracy, self.streaming_accuray_op = \
                tf.contrib.metrics.streaming_accuracy(
                    predictions=self.prediction,
                    labels=self.label,
                    name='streaming_accuracy')

            correct_at_1 = tf.nn.in_top_k(self.logits, self.label, 1)
            self.accuracy_at_1 =\
                tf.reduce_mean(tf.cast(correct_at_1, tf.float32))

        loss_summary = tf.summary.scalar("cost", self.cost)
        streaming_accuray_summary =\
            tf.summary.scalar("streaming_accuray",
                              self.streaming_accuray_op)
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
