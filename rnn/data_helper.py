#!/usr/lib/env python
# -*- code:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import collections
import csv
import numpy as np

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


def create_vocabulary():
    pass