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
"""Example of Estimator for DNN-based text classification with DBpedia data."""

import argparse
import sys
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib
from tensorflow.contrib.learn.python.learn.metric_spec import MetricSpec
import data_helper
import numpy as np

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)


def rnn_model(features, target, mode, vocabulary_size, embedding_size, n_class):
    """RNN model to predict from sequence of words to a class."""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length,
    # EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(
        features, vocab_size=vocabulary_size, embed_dim=embedding_size, scope='words')

    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.contrib.rnn.GRUCell(embedding_size)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic
    # regression over output classes.
    target = tf.one_hot(target, n_class, 1, 0)
    logits = tf.contrib.layers.fully_connected(encoding, n_class, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=target, logits=logits)
    # Create a training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=FLAGS.learning_rate)
    predictions = {
        'classes': tf.argmax(logits, 1),
        'probabilities': tf.nn.softmax(logits, name="softmax_tensor")
    }
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def main(unused_argv):
    # Prepare training and testing data
    x_train, y_train, x_test, y_test, vocabulary_processor = \
        data_helper.load_data_labels(FLAGS.data_dir + FLAGS.data_file, FLAGS.dev_sample_percentage)
    n_class = y_train.shape[1]
    y_test = np.argmax(y_test, 1)
    y_train = np.argmax(y_train, 1)
    # Build model
    classifier = learn.Estimator(model_fn=lambda features, target, mode: rnn_model(features, target, mode, len(
                  vocabulary_processor.vocabulary_), FLAGS.embedding_size, n_class),
                                                model_dir=FLAGS.log_dir)
    # config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1e3)))
    # Train and evaluate
    y_train = (y for y in y_train)
    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #     tensors=tensors_to_log, every_n_iter=100)
    # classifier.fit(x=x_train, y=y_train, batch_size=FLAGS.batch_size, steps=FLAGS.train_steps, monitors=[logging_hook])
    # , monitors=[validation_monitor])
    classifier.fit(x=x_train, y=y_train, batch_size=FLAGS.batch_size, steps=FLAGS.train_steps)
    y_test = (y for y in y_test)
    # Configure the accuracy metric for evaluation
    # metrics = {
    #   learn.metric_spec.MetricSpec(
    #           metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    #                 }
    # score = classifier.evaluate(x=x_test, y=y_test, batch_size=FLAGS.batch_size, steps=FLAGS.dev_steps, metrics=metrics)
    score = classifier.evaluate(x=x_test, y=y_test, batch_size=FLAGS.batch_size, steps=FLAGS.dev_steps)
    print(score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        default="../../data/data_by_ocean/eclipse/",
        help='data direction',
        action='store_true'
    )
    parser.add_argument(
        '--data_file',
        default="sort-text-id.csv",
        help='data path',
        action='store_true'
    )
    parser.add_argument(
        '--log_dir',
        default="rnn_model/",
        help='log direction',
        action='store_true'
    )
    parser.add_argument(
        '--label_file',
        default="fixer.csv",
        help='label path',
        action='store_true'
    )
    parser.add_argument(
        '--embedding_size',
        default=128,
        help='vocabulary size',
        action='store_true'
    )
    parser.add_argument(
        '--batch_size',
        default=10,
        help='batch size',
        action='store_true'
    )
    parser.add_argument(
        '--train_steps',
        default=2e5,
        help='training steps',
        action='store_true'
    )
    parser.add_argument(
        '--dev_steps',
        default=1e5,
        help='evaluating steps',
        action='store_true'
    )
    parser.add_argument(
        '--dev_sample_percentage',
        default=0.2,
        help='Percentage of the training data to use for test',
        action='store_true'
    )
    parser.add_argument(
        '--learning_rate',
        default=1e-2,
        help='learning rate',
        action='store_true'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
