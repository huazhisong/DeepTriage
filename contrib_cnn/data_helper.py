#!/usr/lib/env python
# -*- code:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from tensorflow.contrib import learn
import tensorflow as tf

from tensorflow.python.platform import gfile

Dataset = collections.namedtuple('Dataset', ['data', 'target'])
Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])


def load_data(data_file, label_file):
    """Load dataset from CSV file without a header row."""
    with gfile.Open(data_file) as data:
        with gfile.Open(label_file) as labels:
            lines_data = csv.reader(data)
            lines_labels = csv.reader(labels)
            data, target = [], []
            for d, l in zip(lines_data, lines_labels):
                target.append(l)
                data.append(d)
    target = np.array(target)
    data = np.array(data)
    return Dataset(data=data, target=target)


def load_data_labels(data_file, dev_sample_percentage=0.2):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """

    data = pd.read_csv(data_file, encoding='latin-1')
    x = data.text
    y = data.fixer
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
    x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
    y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

    # 处理training data
    # document length取90%的分位数
    document_length_df = pd.DataFrame([len(xx.split(" ")) for xx in x_train])
    document_length = np.int64(document_length_df.quantile(0.8))
    tf.summary.scalar("document_len", document_length)
    vocabulary_processor = learn.preprocessing.VocabularyProcessor(document_length)
    x_train = vocabulary_processor.fit_transform(x_train)
    x_dev = vocabulary_processor.transform(x_dev)

    # 处理label
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_dev = lb.transform(y_dev)
    print("Document length: %d" % document_length)
    print("Vocabulary Size: {:d}".format(len(vocabulary_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    return x_train, y_train, x_dev, y_dev, vocabulary_processor
