import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelBinarizer
from tensorflow.contrib import learn
import tensorflow as tf
import pdb
import collections


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def load_data_labels(data_files, labels_files, test_data_files, test_labels_files):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    data = []
    for data_file in data_files:
        with open(data_file, 'r', encoding='latin-1') as f:
            data.extend([s.strip() for s in f.readlines()])
            data = [clean_str(s) for s in data]
    print('train data length: %d' % len(data))
    test_data = []
    for test_data_file in test_data_files:
        with open(test_data_file, 'r', encoding='latin-1') as f:
            test_data.extend([s.strip() for s in f.readlines()])
            test_data = [clean_str(s) for s in test_data]
    print('test data length: %d' % len(test_data))

    labels_dfs = [pd.read_csv(f) for f in labels_files]
    labels = pd.concat(labels_dfs)

    labels_dfs = [pd.read_csv(f) for f in test_labels_files]
    test_labels = pd.concat(labels_dfs)

    lb = LabelBinarizer()
    y_train = lb.fit_transform(labels.who)
    y_test = lb.transform(test_labels.who)

    # document length
    document_length_df = pd.DataFrame([len(xx.split(" ")) for xx in data])
    document_length = np.int64(document_length_df.quantile(0.8))
    vocab_processor = learn.preprocessing.VocabularyProcessor(document_length)
    x_train = np.array(list(vocab_processor.fit_transform(data)))
    x_test = np.array(list(vocab_processor.transform(test_data)))
    return x_train, y_train, x_test, y_test, vocab_processor


def load_data_labels(data_file, dev_sample_percentage = 0.2):
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
    vocabulary_processor = learn.preprocessing.VocabularyProcessor(document_length)
    tf.summary.scalar("document_len", document_length)
    x_train = np.array(list(vocabulary_processor.fit_transform(x_train)), dtype=np.float32)
    x_dev = np.array(list(vocabulary_processor.transform(x_dev)))

    # 处理label
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_dev = lb.transform(y_dev)

    print("Vocabulary Size: {:d}".format(len(vocabulary_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    return x_train, y_train, x_dev, y_dev, vocabulary_processor


def batch_iter(data, batch_size, num_epochs=1, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


if __name__ == "__main__":
    x, y, vocab_processor = load_data_labels("../../data/data_by_ocean/eclipse/textForLDA_final.csv",
                                             "../../data/data_by_ocean/eclipse/fixer.csv")
