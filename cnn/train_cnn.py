#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
import text_cnn

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("data_file", "../../data/data_by_ocean/eclipse/sort-text-id.csv",
                       "Data source for the  data.")
tf.flags.DEFINE_string("label_file", "../../data/data_by_ocean/eclipse/fixer.csv", "Data source for the labels data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 12, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1,2", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 12, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.2, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5000, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
# ocean's training data
# x, y, vocab_processor = data_helpers.load_data_labels(FLAGS.data_file, FLAGS.label_file)

# xiaowan training data
x_train, y_train, x_dev, y_dev, vocabulary_processor = data_helpers.load_data_labels(FLAGS.data_file, FLAGS.dev_sample_percentage)
# mine training data
# train_data = ['../../data/data_by_ocean/eclipse/raw/0_summary_description.csv',
#               '../../data/data_by_ocean/eclipse/raw/1_summary_description.csv',
#               '../../data/data_by_ocean/eclipse/raw/2_summary_description.csv',
#               '../../data/data_by_ocean/eclipse/raw/3_summary_description.csv',
#               '../../data/data_by_ocean/eclipse/raw/4_summary_description.csv',
#               '../../data/data_by_ocean/eclipse/raw/5_summary_description.csv',
#               '../../data/data_by_ocean/eclipse/raw/6_summary_description.csv',
#               '../../data/data_by_ocean/eclipse/raw/7_summary_description.csv',
#               '../../data/data_by_ocean/eclipse/raw/8_summary_description.csv']
# label_data = ['../../data/data_by_ocean/eclipse/raw/0_bug_id_date_who.csv',
#               '../../data/data_by_ocean/eclipse/raw/1_bug_id_date_who.csv',
#               '../../data/data_by_ocean/eclipse/raw/2_bug_id_date_who.csv',
#               '../../data/data_by_ocean/eclipse/raw/3_bug_id_date_who.csv',
#               '../../data/data_by_ocean/eclipse/raw/4_bug_id_date_who.csv',
#               '../../data/data_by_ocean/eclipse/raw/5_bug_id_date_who.csv',
#               '../../data/data_by_ocean/eclipse/raw/6_bug_id_date_who.csv',
#               '../../data/data_by_ocean/eclipse/raw/7_bug_id_date_who.csv',
#               '../../data/data_by_ocean/eclipse/raw/8_bug_id_date_who.csv']
# test_data = ['../../data/data_by_ocean/eclipse/raw/9_summary_description.csv',
#              '../../data/data_by_ocean/eclipse/raw/10_summary_description.csv']
# label_test_data = ['../../data/data_by_ocean/eclipse/raw/9_bug_id_date_who.csv',
#                    '../../data/data_by_ocean/eclipse/raw/10_bug_id_date_who.csv']
# x_train, y_train, x_dev, y_dev, vocab_processor = data_helpers.load_data_labels(train_data, label_data,
#                                                                                 test_data, label_test_data)

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
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        # timestamp = str(int(time.time()))
        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs/cnn_model"))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)
        precision_summary = tf.summary.scalar("precision", cnn.precision)
        recall_summary = tf.summary.scalar("recall", cnn.recall)
        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged, precision_summary, recall_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary, precision_summary, precision_summary, recall_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocabulary_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, loss, accuracy, precision, recall = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, pre {:g}, rcl {:g}".format(time_str, step, loss, accuracy, precision, recall))
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy, precision, recall = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.precision, cnn.recall],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}, prc {:g}, rcl {:g}".format(time_str, step, loss, accuracy, precision, recall))
            if writer:
                writer.add_summary(summaries, step)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_batches = data_helpers.batch_iter(list(zip(x_dev, y_dev)), FLAGS.batch_size)

                for dev_batch in dev_batches:
                    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    dev_step(x_dev_batch, y_dev_batch,writer=dev_summary_writer)
                print("")
                # dev_step(x_dev, y_dev, writer=dev_summary_writr)
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))
