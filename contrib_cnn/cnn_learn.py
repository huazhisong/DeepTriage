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
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import tensorflow as tf

FLAGS = None

WORDS_FEATURE = 'words'


def print_operation(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def conception_layer(net, filter_lists, kw, n_filters, document_length, pooling_stride):
    outputs = []

    for window_size in filter_lists:
        print_operation(net)
        with tf.name_scope("convolution_%s" % window_size):
            conv = tf.layers.conv2d(
                net,
                filters=n_filters,
                kernel_size=[window_size, kw],
                padding='VALID',
                # Add a ReLU for non linearity.
                activation=tf.nn.relu)
            # Max pooling across output of Convolution+Relu.
            print_operation(conv)
            pooling = tf.layers.max_pooling2d(
                conv,
                pool_size=(document_length - window_size + 1, 1),
                strides=pooling_stride,
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


def get_pretrained_embedding(vocabulary_processor, embedding_file, n_words, embedding_size):
    print("Load word2vec file {}\n".format(embedding_file))
    initW = np.random.uniform(-0.25, 0.25, (n_words, embedding_size))
    word_vectors = KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    for word in word_vectors.vocab:
        idx = vocabulary_processor.vocabulary_.get(word)
        if idx != 0:
            initW[idx] = word_vectors[word]
    return initW


def cnn_model(features, labels, mode, params):
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    with tf.name_scope("embedding"):
        word_vectors = None
        if params['embedding_type'] == 'random':
            word_vectors = tf.contrib.layers.embed_sequence(
                features[WORDS_FEATURE], vocab_size=params['n_words'], embed_dim=params['embedding_size'])
            word_vectors = tf.expand_dims(word_vectors, -1)
        elif params['embedding_type'] == 'static':
            word_vectors = tf.Variable(tf.constant(0.0, shape=[params['n_words'], params['embedding_size']]),
                                       trainable=False, name="W")
            word_vectors = word_vectors.assign(params['pretrained_embedding'])
            word_vectors = tf.nn.embedding_lookup(word_vectors, features[WORDS_FEATURE])
            word_vectors = tf.expand_dims(word_vectors, -1)
        elif params['embedding_type'] == 'train_static':
            word_vectors = tf.Variable(tf.constant(0.0, shape=[params['n_words'], params['embedding_size']]),
                                       trainable=True, name="W")
            word_vectors = word_vectors.assign(params['pretrained_embedding'])
            word_vectors = tf.nn.embedding_lookup(word_vectors, features[WORDS_FEATURE])
            word_vectors = tf.expand_dims(word_vectors, -1)
        elif params['embedding_type'] == 'multiple_static':
            static_embedding = tf.Variable(tf.constant(0.0, shape=[params['n_words'], params['embedding_size']]),
                                           trainable=False, name="W")
            static_embedding = static_embedding.assign(params['pretrained_embedding'])
            static_words = tf.nn.embedding_lookup(static_embedding, features[WORDS_FEATURE])
            static_words = tf.expand_dims(static_words, -1)
            none_static_embedding = tf.Variable(tf.constant(0.0, shape=[params['n_words'], params['embedding_size']]),
                                                trainable=True, name="W")
            none_static_embedding = none_static_embedding.assign(params['pretrained_embedding'])
            none_static_words = tf.nn.embedding_lookup(none_static_embedding, features[WORDS_FEATURE])
            none_static_words = tf.expand_dims(none_static_words, -1)
            word_vectors = tf.concat([static_words, none_static_words], 3)
        else:
            print("embedding type error")
            return
    with tf.name_scope("layer_1"):
        net = conception_layer(
            word_vectors,
            params['filter_list'],
            params['embedding_size'],
            params['n_filters'],
            params['document_length'],
            params['pooling_stride']
        )
        print_operation(net)

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.name_scope("dropout"):
            net = tf.layers.dropout(net, params['drop_out'])

    with tf.name_scope("output"):
        # Apply regular WX + B and classification
        if params['l2_loss']:
            logits = tf.layers.dense(
                net,
                params['max_label'],
                kernel_regularizer=tf.contrib.layers.l2_regularizer(params['l2_loss']),
                bias_regularizer=tf.contrib.layers.l2_regularizer(params['l2_loss']),
                activation=None)
        else:
            logits = tf.layers.dense(
                net, params['max_label'], activation=None)
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

    one_hot_labels = tf.one_hot(labels, params['max_label'], 1, 0)
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
            labels=labels, predictions=logits, k=params['top_k']),
        'average_precision_k': tf.metrics.sparse_average_precision_at_k(
            labels=labels, predictions=logits, k=params['top_k']),
        'precision_k': tf.metrics.sparse_precision_at_k(
            labels=labels, predictions=logits, k=params['top_k'])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Prepare training and testing data
    dbpedia = tf.contrib.learn.datasets.load_dataset(
        'dbpedia', test_with_fake_data=FLAGS.test_with_fake_data)
    x_train = pandas.DataFrame(dbpedia.train.data)[1]
    y_train = pandas.Series(dbpedia.train.target)
    x_test = pandas.DataFrame(dbpedia.test.data)[1]
    y_test = pandas.Series(dbpedia.test.target)

    # Process vocabulary
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(100)
    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_test = np.array(list(vocab_processor.transform(x_test)))
    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    embedding_file = '../../data/data_by_ocean/GoogleNews-vectors-negative300.bin'
    embedding_size = 300
    document_length = 100
    n_filters = 10
    l2_loss = 3.0
    drop_out = 0.5
    top_k = 1
    filter_list = [3, 4, 5]
    pooling_stride = 1
    max_label = 15
    embedding_type = 'random'
    if embedding_type != 'random':
        pretrained_embedding = get_pretrained_embedding(
            vocab_processor,
            embedding_file,
            n_words,
            embedding_size
        )
    else:
        pretrained_embedding = None

    params = {'document_length': document_length,
              'embedding_size': embedding_size,
              'n_filters': n_filters,
              'n_words': n_words,
              'l2_loss': l2_loss,
              'drop_out': drop_out,
              'top_k': top_k,
              'filter_list': filter_list,
              'embedding_type': embedding_type,
              'vocab_processor': vocab_processor,
              'pretrained_embedding': pretrained_embedding,
              'pooling_stride': pooling_stride,
              'max_label': max_label}
    # Build model
    classifier = tf.contrib.learn.Estimator(model_fn=cnn_model, params=params)
    grid_params = {'l2_loss': [1.0, 2.0]}
    print(classifier.get_params().keys())
    grid_searcher = GridSearchCV(classifier, grid_params, scoring='accuracy', fit_params={'steps': [200, 400]})
    grid_searcher.fit(x_train, y_train)
    score = accuracy_score(y_test, grid_searcher.predict(x_test))
    print(score)
    # # Train.
    # train_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={WORDS_FEATURE: x_train},
    #     y=y_train,
    #     batch_size=len(x_train),
    #     num_epochs=None,
    #     shuffle=True)
    # classifier.train(input_fn=train_input_fn, steps=100)
    #
    # # Predict.
    # test_input_fn = tf.estimator.inputs.numpy_input_fn(
    #     x={WORDS_FEATURE: x_test},
    #     y=y_test,
    #     num_epochs=1,
    #     shuffle=False)
    # predictions = classifier.predict(input_fn=test_input_fn)
    # y_predicted = np.array(list(p['class'] for p in predictions))
    # y_predicted = y_predicted.reshape(np.array(y_test).shape)
    #
    # # Score with sklearn.
    # score = metrics.accuracy_score(y_test, y_predicted)
    # print('Accuracy (sklearn): {0:f}'.format(score))
    # recall = metrics.recall_score(y_test, y_predicted, average='weighted')
    # print('Recall (sklearn): {0:f}'.format(recall))
    # precision = metrics.precision_score(y_test, y_predicted, average='micro')
    # print('precision (sklearn): {0:f}'.format(precision))
    #
    # # Score with tensorflow.
    # scores = classifier.evaluate(input_fn=test_input_fn)
    # print('Accuracy (tensorflow): {0:f}'.format(scores['accuracy']))
    # print('recall_k (tensorflow): {0:f}'.format(scores['recall_k']))
    # print('precision_k (tensorflow): {0:f}'.format(scores['precision_k']))
    # print('average_precision_k (tensorflow): {0:f}'.format(scores['average_precision_k']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--test_with_fake_data',
        default=False,
        help='Test the example code with fake data.',
        action='store_true')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
