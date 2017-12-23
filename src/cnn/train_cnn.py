#! /usr/bin/env python
# -*- coding:utf-8 -*-
from tensorflow.contrib.tensorboard.plugins import projector
from gensim.models.word2vec import KeyedVectors
import tensorflow as tf
import os
import numpy as np
import datetime
import data_helpers
import text_cnn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", 0.1,
                      "Percentage of the training data to use for validation")
tf.flags.DEFINE_string(
    "data_file", "../../data/data_by_ocean/eclipse/sort-text-id.csv",
    "Data source for the  data.")
# tf.flags.DEFINE_string("embedding_file",
# "../../data/data_by_ocean/GoogleNews-vectors-negative300.bin",
#                        "embedding file")
# GoogleNews-vectors-negative300.bin",
tf.flags.DEFINE_string(
    "embedding_file",
    "../../data/data_by_ocean/eclipse.bin",
    "embedding file")
tf.flags.DEFINE_string("checkpointDir", "./runs/cnn_model", "log dir")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300,
                        "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5",
                       "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer(
    "num_filters", 300, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5,
                      "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0,
                      "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("init_learning_rate", 1e-4, "learning rate")
tf.flags.DEFINE_float("decay_rate", 0.96, "decay rate")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size (default: 64)")
tf.flags.DEFINE_integer(
    "num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer(
    "require_improvement", 5,
    "Require improvement steps for training data (default: 1000)")
tf.flags.DEFINE_integer(
    "evaluate_every", 500,
    "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 10000,
                        "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 50000,
                        "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("top_k", 1, "evaluation top k")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False,
                        "Log placement of ops on devices")
tf.flags.DEFINE_string(
    "embedding_type", "non_static",
    "rand, static,non_static, multiple_channels (default: 'rand')")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

if tf.gfile.Exists(FLAGS.checkpointDir):
    tf.gfile.DeleteRecursively(FLAGS.checkpointDir)
tf.gfile.MakeDirs(FLAGS.checkpointDir)

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
train_data = "eclipse/song_no_select/"
train_index = 1
data_dir = "../../data/data_by_ocean/" + train_data
models = "TextCNN"
data_results = data_dir + "results/" + models + '/'
if not tf.gfile.Exists(data_results):
    tf.gfile.MakeDirs(data_results)
class_file = data_results + "class_" + str(train_index) + ".csv"
train_files = [data_dir + str(i) + '.csv' for i in range(train_index)]
test_files = [data_dir +
              str(i) + '.csv' for i in range(train_index, train_index + 1)]
x_train, y_train, x_dev, y_dev, vocabulary_processor = data_helpers.load_files(
    train_files, test_files, class_file)


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = text_cnn.TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocabulary_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            num_filters=FLAGS.num_filters,
            batch_size=FLAGS.batch_size,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            top_k=FLAGS.top_k,
            embedding_type=FLAGS.embedding_type,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # add decay learning rate
        # num_batches_per_epoch =
        # int((len(x_train) - 1) / FLAGS.batch_size) + 1
        # decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs * 0.1)
        # Decay the learning rate exponentially based on the number of steps.
        # lr = tf.train.exponential_decay(FLAGS.init_learning_rate,
        #                                 global_step,
        #                                 decay_steps,
        #                                 FLAGS.decay_rate,
        #                                 staircase=True)
        lr = FLAGS.init_learning_rate
        optimizer = tf.train.AdamOptimizer(lr)
        # optimizer = tf.train.AdadeltaOptimizer(lr)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram(
                    "{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar(
                    "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        streaming_accuray_summary =\
            tf.summary.scalar("streaming_accuray", cnn.streaming_accuray_op)
        accuracy_summary_at_1 = \
            tf.summary.scalar("accuracy_at_1", cnn.accuracy_at_1)
        precision_summary_at_1 = \
            tf.summary.scalar("precision_at_1", cnn.precision_op_at_1)
        recall_summary_at_1 = \
            tf.summary.scalar("recall_at_1", cnn.recall_op_at_1)
        accuracy_summary_at_2 = \
            tf.summary.scalar("accuracy_at_2", cnn.accuracy_at_2)
        precision_summary_at_2 = \
            tf.summary.scalar("precision_at_2", cnn.precision_op_at_2)
        recall_summary_at_2 = \
            tf.summary.scalar("recall_at_2", cnn.recall_op_at_2)
        accuracy_summary_at_3 = \
            tf.summary.scalar("accuracy_at_3", cnn.accuracy_at_3)
        precision_summary_at_3 = \
            tf.summary.scalar("precision_at_3", cnn.precision_op_at_3)
        recall_summary_at_3 = \
            tf.summary.scalar("recall_at_3", cnn.recall_op_at_3)
        accuracy_summary_at_4 = \
            tf.summary.scalar("accuracy_at_4", cnn.accuracy_at_4)
        precision_summary_at_4 = \
            tf.summary.scalar("precision_at_4", cnn.precision_op_at_4)
        recall_summary_at_4 = \
            tf.summary.scalar("recall_at_4", cnn.recall_op_at_4)
        accuracy_summary_at_5 = \
            tf.summary.scalar("accuracy_at_5", cnn.accuracy_at_5)
        precision_summary_at_5 = \
            tf.summary.scalar("precision_at_5", cnn.precision_op_at_5)
        recall_summary_at_5 = \
            tf.summary.scalar("recall_at_5", cnn.recall_op_at_5)

        # Train Summaries
        train_summary_op = tf.summary.merge(
            [loss_summary,
             accuracy_summary_at_1,
             accuracy_summary_at_2,
             accuracy_summary_at_3,
             accuracy_summary_at_4,
             accuracy_summary_at_5,
             grad_summaries_merged])
        train_summary_dir = os.path.abspath(
            os.path.join(FLAGS.checkpointDir, "summaries", "train"))
        train_summary_writer = tf.summary.FileWriter(
            train_summary_dir, sess.graph)
        # test summaries
        test_summary_op = tf.summary.merge(
            [loss_summary,
             streaming_accuray_summary,
             precision_summary_at_1,
             precision_summary_at_2,
             precision_summary_at_3,
             precision_summary_at_4,
             precision_summary_at_5,
             recall_summary_at_1,
             recall_summary_at_2,
             recall_summary_at_3,
             recall_summary_at_4,
             recall_summary_at_5])
        test_summary_dir = os.path.abspath(
            os.path.join(FLAGS.checkpointDir, "summaries", "test"))
        test_summary_writer = tf.summary.FileWriter(
            test_summary_dir, sess.graph)

        # Checkpoint directory. TensorFlow assumes this directory
        # already exists so we need to create it
        checkpoint_dir = os.path.abspath(
            os.path.join(FLAGS.checkpointDir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(
            var_list=tf.trainable_variables(),
            max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        # vocabulary_processor.save(os.path.join(FLAGS.checkpointDir, "vocab"))

        initW = None
        if FLAGS.embedding_type in \
                ['static', 'none_static', 'multiple_channels']:
            # initial matrix with random uniform
            initW = np.random.uniform(
                -0.25,
                0.25,
                (len(vocabulary_processor.vocabulary_), FLAGS.embedding_dim))
            # load any vectors from the word2vec
            print("Load word2vec file {}\n".format(FLAGS.embedding_file))
            word_vectors = KeyedVectors.load_word2vec_format(
                FLAGS.embedding_file, binary=True)
            for word in word_vectors.vocab:
                idx = vocabulary_processor.vocabulary_.get(word)
                if idx != 0:
                    initW[idx] = word_vectors[word]
            sess.run(cnn.W.assign(initW))
            if FLAGS.embedding_type == 'multiple_channels':
                sess.run(cnn.W_static.assign(initW))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        def train_step(x_batch_train, y_batch_train):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch_train,
                cnn.input_y: y_batch_train,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step_train, summaries, loss, \
                acc_at_1, acc_at_2, acc_at_3, acc_at_4, acc_at_5 \
                = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss,
                     cnn.accuracy_at_1,
                     cnn.accuracy_at_2,
                     cnn.accuracy_at_3,
                     cnn.accuracy_at_4,
                     cnn.accuracy_at_5],
                    feed_dict)
            if step_train % FLAGS.evaluate_every == 0:
                time_str = datetime.datetime.now().isoformat()
                print(
                    "{}: step {}, loss {:g}, acc_1 {:g}, \
                        acc_2 {:g}, acc_3 {:g}, acc_4 {:g},\
                         acc_5 {:g}, ".format(time_str,
                                              step_train,
                                              loss,
                                              acc_at_1,
                                              acc_at_2,
                                              acc_at_3,
                                              acc_at_4,
                                              acc_at_5))
                train_summary_writer.add_summary(summaries, step_train)
            return acc_at_1

        def test_step(x_batch_test, y_batch_test, step_test, writer=None):
            """
             Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch_test,
                cnn.input_y: y_batch_test,
                cnn.dropout_keep_prob: 1.0
            }
            summaries, loss,\
                correct_at_1, correct_at_2,\
                correct_at_3, correct_at_4, correct_at_5,\
                streaming_accuray,\
                precision_at_1, precision_at_2, \
                precision_at_3, precision_at_4, precision_at_5, \
                recall_at_1, recall_at_2,\
                recall_at_3, recall_at_4, recall_at_5,\
                prediction_top_k_indices,\
                prediction_top_k_values = \
                sess.run([test_summary_op,
                          cnn.loss,
                          cnn.correct_at_1,
                          cnn.correct_at_2,
                          cnn.correct_at_3,
                          cnn.correct_at_4,
                          cnn.correct_at_5,
                          cnn.streaming_accuray_op,
                          cnn.precision_op_at_1,
                          cnn.precision_op_at_2,
                          cnn.precision_op_at_3,
                          cnn.precision_op_at_4,
                          cnn.precision_op_at_5,
                          cnn.recall_op_at_1,
                          cnn.recall_op_at_2,
                          cnn.recall_op_at_3,
                          cnn.recall_op_at_4,
                          cnn.recall_op_at_5,
                          cnn.prediction_top_k_indices,
                          cnn.prediction_top_k_values],
                         feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, streaming_accuracy {:g},\
                prc_1 {:g}, prc_2 {:g}, prc_3 {:g}, prc_4 {:g}, prc_5 {:g}, \
                rcl_1 {:g}, rcl_2 {:g}, rcl_3 {:g}, rcl_4 {:g}, rcl_5 {:g}, ".
                  format(time_str,
                         step_test, loss,
                         streaming_accuray,
                         precision_at_1,
                         precision_at_2,
                         precision_at_3,
                         precision_at_4,
                         precision_at_5,
                         recall_at_1,
                         recall_at_2,
                         recall_at_3,
                         recall_at_4,
                         recall_at_5))

            if writer:
                writer.add_summary(summaries, step_test)
            return prediction_top_k_values,\
                prediction_top_k_indices,\
                (np.sum(correct_at_1), np.sum(correct_at_2), np.sum(correct_at_3), np.sum(correct_at_4), np.sum(correct_at_5))

        # Generate batches
        batches = data_helpers.batch_generator(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        best_accuracy = 0.0
        last_improvement_step = 0
        numer_iter = int((len(y_train) - 1) / FLAGS.batch_size) + 1
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            accuracy = train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix,
                                  global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                last_improvement_step = current_step
            if (current_step - last_improvement_step > (FLAGS.require_improvement * numer_iter)) or  best_accuracy == 1.0:
                print('no more improving!')
                break
        current_step = tf.train.global_step(sess, global_step)
        path = saver.save(sess, checkpoint_prefix,
                          global_step=current_step)
        print("Saved newest model checkpoint to {}\n".format(path))
        # embedding summaries
        summary_dir = os.path.abspath(os.path.join(FLAGS.checkpointDir))
        summary_writer = tf.summary.FileWriter(test_summary_dir)
        # projector embedding
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = cnn.W.name
        embedding.metadata_path = os.path.abspath(
            os.path.join(FLAGS.checkpointDir, 'metadata.tsv'))
        projector.visualize_embeddings(summary_writer, config)

        print("\n Testing:")
        dev_batches = data_helpers.batch_generator(
            list(zip(x_dev, y_dev)), FLAGS.batch_size)
        step = 0
        prediction_top_k_values = []
        prediction_top_k_indices = []
        real_labels_indice = []
        true_correct = np.zeros(5)
        for dev_batch in dev_batches:
            x_dev_batch, y_dev_batch = zip(*dev_batch)
            values, indices, correct = test_step(
                x_dev_batch, y_dev_batch, step, writer=test_summary_writer)
            true_correct += correct
            prediction_top_k_values.extend(values)
            prediction_top_k_indices.extend(indices)
            real_labels_indice.extend(np.argmax(y_dev_batch, 1))
            step += 1

        numer_iter = int((len(y_dev) - 1) / FLAGS.batch_size) + 1
        total_nums = numer_iter * FLAGS.batch_size
        for k in range(5):
            print('%s: total accuracy @ %d = %.8f' %
                  (datetime.datetime.now().isoformat(),
                   k + 1,
                   (true_correct[k] / total_nums)))
        fixer_file = data_results + "fixer_" + str(train_index) + ".csv"
        indices_file = data_results + "prediction_" +\
            str(train_index) + ".csv"
        values_file = data_results + "probability_" + str(train_index) + ".csv"
        np.savetxt(fixer_file, real_labels_indice, fmt="%s", delimiter=',')
        np.savetxt(
            indices_file, prediction_top_k_indices, fmt="%s", delimiter=',')
        np.savetxt(
            values_file, prediction_top_k_values, fmt="%s", delimiter=',')
