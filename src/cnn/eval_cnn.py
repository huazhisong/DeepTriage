#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn
import csv

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("data_file", "../../data/data_by_ocean/eclipse/sort-text-id.csv",
                       "Data source for the  data.")
# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 1024, "Batch Size (default: 64)")
tf.flags.DEFINE_string("--log_dir", "./runs/cnn_model", "log dir")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
tf.flags.DEFINE_float("dev_sample_percentage", .2,
                      "Percentage of the training data to use for validation")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# CHANGE THIS: Load data. Load your own data here
_, _, x_test, y_test, _ = \
    data_helpers.load_data_labels(FLAGS.data_file, FLAGS.dev_sample_percentage)

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(
    os.path.abspath(os.path.join(FLAGS.log_dir, "checkpoints")))
graph = tf.Graph()
with graph.as_default() as g:
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    sess.run(tf.local_variables_initializer())
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        # precision_op = graph.get_operation_by_name("evaluation/precision/update")
        precision_op = graph.get_operation_by_name("evaluation/precision/update").outputs[0]
        precision = graph.get_operation_by_name("evaluation/precision").outputs[0]
        recall_op = graph.get_operation_by_name("evaluation/recall/update").outputs[0]
        recall = graph.get_operation_by_name("evaluation/recall").outputs[0]

        precision_summary = tf.summary.scalar("precision", precision)
        recall_summary = tf.summary.scalar("recall", recall)
        summary_op = tf.summary.merge([precision_summary, recall_summary])
        dev_summary_dir = os.path.abspath(os.path.join(FLAGS.log_dir, "summaries", "evaluations"))
        summary_writer = tf.summary.FileWriter(dev_summary_dir, g)
        # Generate batches for one epoch
        batches = data_helpers.batch_iter(
            list(zip(x_test, y_test)), FLAGS.batch_size, 1, shuffle=True)

        # Collect the predictions here
        all_predictions = []
        precision_total = 0.0
        recall_total = 0.0
        for batch in batches:
            x_test_batch, y_test_batch = zip(*batch)
            _, _, summary, precision_b, recall_b = sess.run(
                [summary_op, precision_op, recall_op, precision, recall],
                {input_x: x_test_batch, input_y: y_test, dropout_keep_prob: 1.0})
            summary_writer.add_summary(summary)

        # Print accuracy if y_test is defined
        if y_test is not None:
            print("Total number of test examples: {}".format(len(y_test)))
            print("Precision: {:g}".format(precision_total))
            print("Recall: {:g}".format(recall_total))

        # Save the evaluation to a csv
        predictions_human_readable = np.column_stack((np.array(x_test), all_predictions))
        out_path = os.path.join(FLAGS.log_dir, "summaries", "evaluations", "prediction.csv")
        print("Saving evaluation to {0}".format(out_path))
        with open(out_path, 'w') as f:
            csv.writer(f).writerows(predictions_human_readable)
