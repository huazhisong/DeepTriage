# -*- code:utf-8 -*-
import pandas as pd
import numpy as np
import os
from tensorflow.contrib import learn
from gensim.models.word2vec import KeyedVectors
from sklearn.preprocessing import LabelBinarizer


def load_files(
        data_files, class_file='',
        embedding_file='', validation=False,
        embedding_dim=300, encode='utf-8'):
    # 读取数据
    if validation:
        train_list = data_files[:-2]
        train = pd.concat([pd.read_csv(file, encoding=encode)
                           for file in train_list])

        dev_file = data_files[-2]
        dev_data = pd.read_csv(dev_file, encoding=encode)
        x_dev = dev_data.text
        y_dev = dev_data.fixer
    else:
        train_list = data_files[:-1]
        train = pd.concat([pd.read_csv(file, encoding=encode)
                           for file in train_list])
    x_train = train.text
    y_train = train.fixer
    test_file = data_files[-1]
    test = pd.read_csv(test_file, encoding=encode)
    x_test = test.text
    y_test = test.fixer
    # 处理文本数据
    document_length_df = pd.DataFrame([len(xx.split(" ")) for xx in x_train])
    document_length = np.int64(document_length_df.quantile(0.8))

    vocabulary_processor = \
        learn.preprocessing.VocabularyProcessor(document_length)

    x_train = np.array(list(vocabulary_processor.fit_transform(x_train)))
    x_test = np.array(list(vocabulary_processor.transform(x_test)))
    if validation:
        x_dev = np.array(list(vocabulary_processor.transform(x_dev)))

    # initial matrix with random uniform
    embedding = np.random.uniform(
        -0.25,
        0.25,
        (len(vocabulary_processor.vocabulary_), embedding_dim))
    # load any vectors from the word2vec
    print("Load word2vec file {}\n".format(embedding_file))

    word_vectors = KeyedVectors.load_word2vec_format(
        embedding_file, binary=True)
    for word in word_vectors.vocab:
        idx = vocabulary_processor.vocabulary_.get(word)
        if idx != 0:
            embedding[idx] = word_vectors[word]
    lb = LabelBinarizer()
    lb.fit(y_train)
    np.savetxt(class_file, lb.classes_, fmt="%s")
    if validation:
        return x_train, y_train,\
            x_dev, y_dev, x_test, y_test, embedding, lb
    return x_train, y_train, x_test, y_test, embedding, lb


def batch_generator(
        data, labels, labelBinarizer,
        batch_size, num_epochs=1, shuffle=False):
    """

    :param data:
    :param batch_size:
    :param num_epochs:
    :param shuffle:
    :return:
    """

    # 处理label
    labels_code = labelBinarizer.transform(labels)
    #
    data = np.array(list(zip(data, labels_code)))
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1

    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
            labels = labels[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = (batch_num + 1) * batch_size
            # end_index = min((batch_num + 1) * batch_size, data_size)
            if end_index < data_size:
                yield shuffled_data[start_index:
                                    end_index], labels[start_index: end_index]
            else:
                rest_part = shuffled_data[start_index:]
                rest_labels = labels[start_index: end_index]
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = shuffled_data[shuffle_indices]
                labels = labels[shuffle_indices]
                new_part = shuffled_data[:batch_size -
                                         (data_size - start_index)]
                new_labels = labels[:batch_size -
                                    (data_size - start_index)]
                yield np.concatenate((rest_part, new_part),
                                     axis=0), np.concatenate(
                    (rest_labels, new_labels), axis=0)


if __name__ == '__main__':
    print(os.listdir('../../data/data_by_ocean/eclipse/song_no_select'))
    data_dir = '../../data/data_by_ocean/eclipse/song_no_select/'
    data_files = [data_dir + str(i) + '.csv' for i in range(11)]
    x_train, y_train, x_test,\
        y_test, vocabulary_processor = load_files(data_files)
