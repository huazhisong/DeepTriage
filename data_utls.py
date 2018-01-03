# -*- code:utf-8 -*-
import pandas as pd
import numpy as np
import os
from collections import Counter
from tensorflow.contrib import learn
from gensim.models.word2vec import KeyedVectors
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import SelectKBest


def classification_score(y_true, y_prediction):

    def get_top_k(k=1):
        y_pred = []
        for i in range(len(y_prediction)):
            if y_true[i] in y_prediction[i][:k]:
                y_pred.append(y_true[i])
            else:
                y_pred.append(y_prediction[i][0])
        return y_pred
    counter = Counter(y_true)
    weighted = [counter.get(item) for item in y_true]
    # metrics
    accuracy = ['accuracy']
    recall_weighted = ['recall']
    precision_weighted = ['precision']
    precision_recommend = ['precision_recommend']
    f1_score_weighted = ['f1_score']
    for k in range(1, 16):
        y_pred = get_top_k(k)

        tmp = accuracy_score(y_true, y_pred)
        accuracy.append(tmp)

        tmp = recall_score(y_true, y_pred, average='weighted',
                           sample_weight=weighted)
        recall_weighted.append(tmp)

        tmp = precision_score(y_true, y_pred, average='weighted',
                              sample_weight=weighted)
        precision_weighted.append(tmp)

        tmp = precision_score(y_true, y_pred, average='micro',
                              sample_weight=weighted)
        precision_recommend.append(tmp / k)

        tmp = f1_score(y_true, y_pred, average='weighted',
                       sample_weight=weighted)
        f1_score_weighted.append(tmp)
    return np.stack([accuracy, recall_weighted,
                     precision_weighted, precision_recommend,
                     f1_score_weighted])


def load_files(
        data_files, class_file='',
        embedding_file='', validation=False,
        embedding_dim=300, encode='utf-8',
        featurs_selection='chi2'):
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

    # 选择特征
    vectorizer = TfidfVectorizer()
    vectors_train = vectorizer.fit_transform(x_train)
    features_names = vectorizer.get_feature_names()
    num_features_selected = int(vectors_train.shape[1] * 0.05)
    if featurs_selection in ['chi2', 'mutual_info_classif']:
        selection = SelectKBest(
            eval(featurs_selection), k=num_features_selected)
        selection.fit(vectors_train, y_train)
        features_names_selected =\
            [features_names[k]
             for k in selection.get_support(indices=True)]
    vocabulary = learn.preprocessing.CategoricalVocabulary()
    for feature in features_names_selected:
        vocabulary.add(feature)
    # 处理文本数据
    document_length_df = pd.DataFrame(
        [len(xx.split(" ")) for xx in x_train])
    document_length = np.int64(document_length_df.quantile(0.8))

    vocabulary_processor = \
        learn.preprocessing.VocabularyProcessor(
            document_length, vocabulary=vocabulary)

    x_train = np.array(
        list(vocabulary_processor.fit_transform(x_train)))
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
    # data_dir = '../../data/data_by_ocean/eclipse/song_no_select/'
    # data_files = [data_dir + str(i) + '.csv' for i in range(11)]
    # x_train, y_train, x_test,\
    #     y_test, vocabulary_processor = load_files(data_files)
