from gensim.models.word2vec import KeyedVectors
from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np

def get_embedding(input, pretrain_embedding_path, embedding_type, vocabulary_size, emb_dim, vocab_file_path):

    init_width = 0.5 / emb_dim
    with tf.name_scope('embedding'):
        if embedding_type == 'rand':
            W = tf.Variable(tf.random_uniform([vocabulary_size, emb_dim], -init_width, init_width), trainable=True, name="W")
            embedded_chars = tf.nn.embedding_lookup(W, input)
            return tf.expand_dims(embedded_chars, -1)

        initW = np.random.uniform(-0.25, 0.25, (vocabulary_size, emb_dim))
        # load any vectors from the word2vec
        print("Load word2vec file {}\n".format(pretrain_embedding_path))
        word_vectors = KeyedVectors.load_word2vec_format(pretrain_embedding_path, binary=True)
        vocabulary_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_file_path)
        for word in word_vectors.vocab:
            idx = vocabulary_processor.vocabulary_.get(word)
            if idx != 0:
                initW[idx] = word_vectors[word]

        if embedding_type == 'static':
            W = tf.Variable(tf.random_uniform([vocabulary_size, emb_dim], -init_width, init_width), trainable=False, name="W")
            W.assign(initW)
            embedded_chars = tf.nn.embedding_lookup(W, input)
            return tf.expand_dims(embedded_chars, -1)

        if embedding_type == 'non_static':
            W = tf.Variable(tf.random_uniform([vocabulary_size, emb_dim], -init_width, init_width), trainable=True, name="W")
            W.assign(initW)
            embedded_chars = tf.nn.embedding_lookup(W, input)
            return tf.expand_dims(embedded_chars, -1)


        if embedding_type == 'multiple_channels':
            W = tf.Variable(tf.random_uniform([vocabulary_size, emb_dim], -init_width, init_width), trainable=True, name="W")
            W.assign(initW)
            W_rand = tf.Variable(tf.random_uniform([vocabulary_size, emb_dim], -init_width, init_width), trainable=True, name="W")
            embedded_rand = tf.nn.embedding_lookup(W_rand, input)
            embedded_chars = tf.nn.embedding_lookup(W, input)
            return tf.concat([tf.expand_dims(embedded_rand, -1), tf.expand_dims(embedded_chars, -1)], 3)

def inference(embedded_data, filter_sizes, num_filters, num_classes, dropout_keep_prob):

    embedding_size = embedded_data.shape()[2]
    sequence_length = embedded_data.shape()[1]
    pooled_outputs = []
    for i , filter_size in enumerate(filter_sizes):
        with tf.name_scope('conv_maxpool_%s' %filter_size):
            # convolution layer
            filter_shape = [filter_size, embedding_size, 1, num_filter]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(embedded_data, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
            pooled_outputs.append(pooled)

    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob, name='dropout')
        # Final (unnormalized) scores and predictions
        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)
    
    with tf.name_scope("output"):
        W = tf.get_variable("W", shape=[num_filters_total, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        logits = tf.nn.xw_plus_b(h_drop, W, b, name="logits")

        return logits, l2_loss

def loss(logits, labels, l2_loss, l2_reg_lambda):
    # an cross-entropy los
    with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        tf.summary.scalar("loss", loss)
        return loss

def train(loss, init_learning_rate, global_step):
    #Define Training procedure
    # add decay learnbete
    # num_batches_per_epoch = int((len(x_train) - 1) / FLAGS.batch_size) +
    # decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs * 0.1)
    # Decssay the learning rate exponentially based on the number of steps.d
    # lr = tf.train.exponential_decay(FLAGS.init_learning_rate,
    #                                 global_step,
    #
    #                                 decay_step
    #                                 FLAGS.decay_rate,
    # staircase=True
    lr = init_learning_rate
    # optimizer = tf.trai
    mOptimizer(lr)
    optimizer = tf.train.AdadeltaOptimizer(lr)
    grads_and_vars = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # track of gradient values and sparsity (optional)    
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)

    grad_summaries_merged = tf.summary.merge(grad_summaries)
    return train_op, grad_summaries_merged

def evaluation(logits, labels, top_k):
    label = tf.arg_max(labels, 1)
    # Accuracy
    with tf.name_scope("accuracy"):
        correct = tf.cast(tf.nn.in_top_k(logits, label, top_k))
        accuracy = tf.reduce_mean(correct)
        summary.scalar("accuracy", accuracy)

    # Evaluation
    with tf.name_scope("evaluation"):
        precision_op, precision = tf.contrib.metrics.streaming_sparse_precision_at_k(logits, label, top_k, name="precision")
        tf.summary.scalar("precision", precision)
        recall_op, recall = tf.contrib.metrics.streaming_sparse_recall_at_k(logits, label, top_k, name="recall")
        tf.summary.scalar("recall", cnn.recall)
        return correct, accuracy, precision_op, precision, recall_op, recall