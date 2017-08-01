#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Example of Estimator for CNN-based text classification with DBpedia data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
from gensim.models.word2vec import KeyedVectors

import numpy as np
import pandas
from sklearn import metrics, grid_search
import tensorflow as tf

FLAGS = None

MAX_DOCUMENT_LENGTH = 100
EMBEDDING_SIZE = 300
N_FILTERS = 10
POOLING_WINDOW = 2
POOLING_STRIDE = 1
n_words = 0
MAX_LABEL = 15
WORDS_FEATURE = 'words'  # Name of the input words feature.
l2_loss = 1.0
drop_out = 0.0
k = 1
filter_list = [3, 4, 5]
embedding_type = 'random'
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
    MAX_DOCUMENT_LENGTH)
embedding_file = '../../data/data_by_ocean/GoogleNews-vectors-negative300.bin'
pretrained_embedding = None


def print_operation(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def conception_layer(net, filter_lists, kw, n_filters=100):
    outputs = []

    for window_size in filter_lists:
        print_operation(net)
        with tf.name_scope("convolution_%s" % window_size):
            conv = tf.layers.conv2d(
                net,
                filters=N_FILTERS,
                kernel_size=[window_size, kw],
                padding='VALID',
                # Add a ReLU for non linearity.
                activation=tf.nn.relu)
            # Max pooling across output of Convolution+Relu.
            print_operation(conv)
            pooling = tf.layers.max_pooling2d(
                conv,
                pool_size=(MAX_DOCUMENT_LENGTH - window_size + 1, 1),
                strides=POOLING_STRIDE,
                padding='VALID')
            print_operation(pooling)
            # Transpose matrix so that n_filters from convolution becomes width.
            # pooling = tf.transpose(pooling, [0, 1, 3, 2])
            outputs.append(pooling)
            # print_operation(pooling)
    num_filters_total = n_filters * len(filter_lists)
    net = tf.concat(outputs, 3)
    print_operation(net)
    return tf.reshape(net, [-1, num_filters_total])


def get_pretrained_embedding(vocabulary_processor):
    print("Load word2vec file {}\n".format(embedding_file))
    initW = np.random.uniform(-0.25, 0.25, (n_words, EMBEDDING_SIZE))
    word_vectors = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    for word in word_vectors.vocab:
        idx = vocabulary_processor.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = word_vectors[word]
    return initW


def cnn_model(features, labels, mode):
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    with tf.name_scope("embedding"):
        word_vectors = None
        if embedding_type == 'random':
            word_vectors = tf.contrib.layers.embed_sequence(
                features[WORDS_FEATURE], vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
            word_vectors = tf.expand_dims(word_vectors, -1)
        elif embedding_type == 'static':
            word_vectors = tf.Variable(tf.constant(0.0, shape=[n_words, EMBEDDING_SIZE]),
                                       trainable=False, name="W")
            word_vectors = word_vectors.assign(pretrained_embedding)
            word_vectors = tf.nn.embedding_lookup(word_vectors, features[WORDS_FEATURE])
            word_vectors = tf.expand_dims(word_vectors, -1)
        elif embedding_type == 'train_static':
            word_vectors = tf.Variable(tf.constant(0.0, shape=[n_words, EMBEDDING_SIZE]),
                                       trainable=True, name="W")
            word_vectors = word_vectors.assign(pretrained_embedding)
            word_vectors = tf.nn.embedding_lookup(word_vectors, features[WORDS_FEATURE])
            word_vectors = tf.expand_dims(word_vectors, -1)
        elif embedding_type == 'multiple_static':
            static_embedding = tf.Variable(tf.constant(0.0, shape=[n_words, EMBEDDING_SIZE]),
                                       trainable=False, name="W")
            static_embedding = static_embedding.assign(pretrained_embedding)
            static_words = tf.nn.embedding_lookup(static_embedding, features[WORDS_FEATURE])
            static_words = tf.expand_dims(static_words, -1)
            none_static_embedding = tf.Variable(tf.constant(0.0, shape=[n_words, EMBEDDING_SIZE]),
                                       trainable=True, name="W")
            none_static_embedding = none_static_embedding.assign(pretrained_embedding)
            none_static_words = tf.nn.embedding_lookup(none_static_embedding, features[WORDS_FEATURE])
            none_static_words = tf.expand_dims(none_static_words, -1)
            word_vectors = tf.concat([static_words, none_static_words], 3)
        else:
            print("embedding type error")
            return
    with tf.name_scope("layer_1"):
        net = conception_layer(word_vectors, filter_list, EMBEDDING_SIZE, N_FILTERS)
        print_operation(net)

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.name_scope("dropout"):
            net = tf.layers.dropout(net, drop_out)

    with tf.name_scope("output"):
        # Apply regular WX + B and classification
        if l2_loss:
            logits = tf.layers.dense(
                net,
                MAX_LABEL,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_loss),
                bias_regularizer=tf.contrib.layers.l2_regularizer(l2_loss),
                activation=None)
        else:
            logits = tf.layers.dense(
                net, MAX_LABEL, activation=None)
    print_operation(logits)
    with tf.name_scope("prediction"):
        predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits)
            })

    one_hot_labels = tf.one_hot(labels, MAX_LABEL, 1, 0)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_labels, logits=logits)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    labels = tf.cast(labels, tf.int64)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes),
        'recall_k': tf.metrics.recall_at_k(
            labels=labels, predictions=logits, k=k),
        'average_precision_k': tf.metrics.sparse_average_precision_at_k(
            labels=labels, predictions=logits, k=k),
        'precision_k': tf.metrics.sparse_precision_at_k(
            labels=labels, predictions=logits, k=k)
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    global n_words
    # Prepare training and testing data
    dbpedia = tf.contrib.learn.datasets.load_dataset(
        'dbpedia', test_with_fake_data=FLAGS.test_with_fake_data)
    x_train = pandas.DataFrame(dbpedia.train.data)[1]
    y_train = pandas.Series(dbpedia.train.target)
    x_test = pandas.DataFrame(dbpedia.test.data)[1]
    y_test = pandas.Series(dbpedia.test.target)

    # Process vocabulary
    global vocab_processor
    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_test = np.array(list(vocab_processor.transform(x_test)))
    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    global pretrained_embedding
    pretrained_embedding = get_pretrained_embedding(vocab_processor)

    # Build model
    classifier = tf.estimator.Estimator(model_fn=cnn_model)

    # Train.
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_train},
        y=y_train,
        batch_size=len(x_train),
        num_epochs=None,
        shuffle=True)
    classifier.train(input_fn=train_input_fn, steps=100)

    # Predict.
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={WORDS_FEATURE: x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    predictions = classifier.predict(input_fn=test_input_fn)
    y_predicted = np.array(list(p['class'] for p in predictions))
    y_predicted = y_predicted.reshape(np.array(y_test).shape)

    # Score with sklearn.
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy (sklearn): {0:f}'.format(score))
    recall = metrics.recall_score(y_test, y_predicted, average='weighted')
    print('Recall (sklearn): {0:f}'.format(recall))
    precision = metrics.precision_score(y_test, y_predicted, average='micro')
    print('precision (sklearn): {0:f}'.format(precision))

    # Score with tensorflow.
    scores = classifier.evaluate(input_fn=test_input_fn)
    print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))
    print('recall_k (tensorflow): {0:f}'.format(scores['recall_k']))
    print('precision_k (tensorflow): {0:f}'.format(scores['precision_k']))
    print('average_precision_k (tensorflow): {0:f}'.format(scores['average_precision_k']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_with_fake_data',
        default=False,
        help='Test the example code with fake data.',
        action='store_true')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
