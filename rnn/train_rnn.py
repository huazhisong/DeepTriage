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
import data_helper

FLAGS = None
learn = tf.contrib.learn


def rnn_model(features, target, vocabulary_size, embedding_size, n_class):
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
    # target = tf.one_hot(target, 15, 1, 0)
    logits = tf.contrib.layers.fully_connected(encoding, n_class, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=target, logits=logits)
    # Create a training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=FLAGS.learning_rate)

    return ({
                'class': tf.argmax(logits, 1),
                'prob': tf.nn.softmax(logits)
            }, loss, train_op)


def main(unused_argv):
    # Prepare training and testing data
    x_train, y_train, x_test, y_test, vocabulary_processor = \
        data_helper.load_data_labels(FLAGS.data_dir + FLAGS.data_file, FLAGS.dev_sample_percentage)
    n_class = len(y_train.unique())
    # Build model
    classifier = learn.SKCompat(
        learn.Estimator(model_fn=lambda features, target: rnn_model(features, target,
                                                                    vocabulary_processor.vocabulary_,
                                                                    FLAGS.embedding_size,
                                                                    n_class),
                        model_dir="/tmp/rnn_model"))

    # Train and predict
    classifier.fit(x_train, y_train, steps=FLAGS.train_steps)
    accuracy = classifier.evaluate(x_test, y_test, steps=FLAGS.dev_steps)['accuracy']
    print('Accuracy: {0:f}'.format_map(accuracy))


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
        '--label_file',
        default="fixer.csv",
        help='label path',
        action='store_true'
    )
    parser.add_argument(
        '--embedding_size',
        default=300,
        help='vocabulary size',
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
        default=1e4,
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
        default=1e-4,
        help='vocabulary size',
        action='store_true'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
