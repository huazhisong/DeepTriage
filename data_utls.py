# -*- code:utf-8 -*-
import pandas as pd
import numpy as np
import os
import re
import json
import string
import nltk
from gensim.models import Word2Vec
from collections import Counter
from tensorflow.contrib import learn
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.metrics import precision_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import SelectKBest, SelectPercentile

import prepocessing_bugs


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


def load_files(data_files, encode='latin2', validation=False):
    # 使用同一种数据源进行训练和测试
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

    if validation:
        return x_train, y_train.values, x_dev, y_dev, x_test, y_test
    else:
        return x_train, y_train.values, x_test, y_test


def load_train_test_files(train_files, test_file, encode='utf8'):
    # 使用不同数据源进行训练和测试
    # 读取数据
    train = pd.concat([pd.read_csv(file, encoding=encode)
                       for file in train_files])
    x_train = train.text
    y_train = train.fixer
    test = pd.read_csv(test_file, encoding=encode)
    x_test = test.text
    y_test = test.fixer

    return x_train, y_train.values, x_test, y_test


def features_selection(x_train, y_train, featurs_selection, percent):
    # 选择特征
    vectorizer = TfidfVectorizer()
    vectors_train = vectorizer.fit_transform(x_train)
    features_names = vectorizer.get_feature_names()
    if percent == 1.0:
        return features_names
    # num_features_selected = int(vectors_train.shape[1] * 0.05)
    if featurs_selection in ['chi2', 'mutual_info_classif']:
        selection = SelectPercentile(
            eval(featurs_selection), percentile=int(percent * 100))
        selection.fit(vectors_train, y_train)
        features_names_selected =\
            [features_names[k]
             for k in selection.get_support(indices=True)]

    elif featurs_selection in ['WLLR', 'IG', 'MI']:
        features_names_selected = prepocessing_bugs.feature_selection(
            [doc.split() for doc in x_train],
            y_train, featurs_selection, percent)

    print('sklearn select features: %d' % len(features_names_selected))
    print(features_names_selected[:10])
    return features_names_selected


TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+",
                          re.UNICODE)
reg_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|\
    [$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def precessing(item):
    # mybluemix precessing words

    # 1. Remove \r
    current_title = item['issue_title'].replace('\r', ' ')
    current_desc = item['description'].replace('\r', ' ')
    # 2. Remove URLs
    current_desc = re.sub(
        reg_pattern, '', current_desc)
    # 3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    # 4. Remove hex code
    current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
    current_title = re.sub(r'(\w+)0x\w+', '', current_title)
    # 5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()
    # 6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    # 7. Strip trailing punctuation marks
    current_desc_filter = [word.strip(string.punctuation)
                           for word in current_desc_tokens]
    current_title_filter = [word.strip(string.punctuation)
                            for word in current_title_tokens]
    # 8. Join the lists
    current_data = current_title_filter + current_desc_filter
    current_data = filter(None, current_data)
    return current_data


def precessing_word2vec(
        data_dir,
        data_set,
        min_word_frequency_word2vec=5,
        embed_size_word2vec=200,
        context_window_word2vec=5):
    # mybluemix word2vec training
    open_bugs_json = data_dir + data_set + '/deep_data.json'
    with open(open_bugs_json) as data_file:
        data = json.load(data_file, strict=False)
    all_data = []
    for item in data:
        current_data = precessing(item)
        current_data = filter(None, current_data)
        all_data.append(list(current_data))
    # Learn the word2vec model and extract vocabulary
    wordvec_model = Word2Vec(
        all_data, min_count=min_word_frequency_word2vec,
        size=embed_size_word2vec, window=context_window_word2vec)
    wordvec_model.save_word2vec_format(
        fname=data_dir + data_set + '.bin', binary=True)


def preprocessing_json(
        data_dir, data_set, file,
        numCV=10, max_sentence_len=50,
        min_sentence_length=15, batch_size=32,
        embed_size_word2vec=200):
    # splits json for numCV
    closed_bugs_json = data_dir + data_set + \
        '/train_test_json/' + file + '.json'
    with open(closed_bugs_json) as data_file:
        data = json.load(data_file, strict=False)
    all_data = []
    all_owner = []
    for item in data:
        current_data = precessing(item)
        all_data.append(list(current_data))
        all_owner.append(item['owner'])
    # load any vectors from the word2vec
    embedding_file = data_dir + data_set + '/' + data_set + '.bin'
    print("Load word2vec file {}\n".format(embedding_file))
    word_vectors = KeyedVectors.load_word2vec_format(
        embedding_file, binary=True)
    vocabulary = list(word_vectors.vocab.keys())
    vocabulary.insert(0, 'UNK')
    totalLength = len(all_data)
    splitLength = int((totalLength - 1) / (numCV + 1)) + 1
    for i in range(1, numCV + 1):
        # Split cross validation set
        train_data = all_data[:i * splitLength - 1]
        test_data = all_data[i * splitLength:(i + 1) * splitLength - 1]
        train_owner = all_owner[:i * splitLength - 1]
        test_owner = all_owner[i * splitLength:(i + 1) * splitLength - 1]

        # Remove words outside the vocabulary
        updated_train_data = []
        updated_train_owner = []
        final_test_data = []
        final_test_owner = []
        for j, item in enumerate(train_data):
            current_train_filter = [
                word for word in item if word in vocabulary]
            if len(current_train_filter) >= min_sentence_length:
                updated_train_data.append(current_train_filter)
                updated_train_owner.append(train_owner[j])

        for j, item in enumerate(test_data):
            current_test_filter = [word for word in item if word in vocabulary]
            if len(current_test_filter) >= min_sentence_length:
                final_test_data.append(current_test_filter)
                final_test_owner.append(test_owner[j])

        # Remove data from test set that is not there in train set
        train_owner_unique = set(updated_train_owner)
        test_owner_unique = set(final_test_owner)
        unwanted_owner = list(test_owner_unique - train_owner_unique)
        updated_test_data = []
        updated_test_owner = []
        for j in range(len(final_test_owner)):
            if final_test_owner[j] not in unwanted_owner:
                updated_test_data.append(final_test_data[j])
                updated_test_owner.append(final_test_owner[j])

        # Create train and test data for deep learning + softmax
        X_train = np.empty(
            shape=[len(updated_train_data), max_sentence_len],
            dtype='float32')
        Y_train = updated_train_owner
        for j, curr_row in enumerate(updated_train_data):
            sequence_cnt = 0
            for item in curr_row:
                if item in vocabulary:
                    X_train[j, sequence_cnt] = vocabulary.index(item)
                    sequence_cnt = sequence_cnt + 1
                    if sequence_cnt == max_sentence_len - 1:
                        break
            for k in range(sequence_cnt, max_sentence_len):
                X_train[j, k] = np.zeros((1,))

        X_test = np.empty(
            shape=[len(updated_test_data), max_sentence_len],
            dtype='float32')
        Y_test = updated_test_owner
        for j, curr_row in enumerate(updated_test_data):
            sequence_cnt = 0
            for item in curr_row:
                if item in vocabulary:
                    X_test[j, sequence_cnt] = vocabulary.index(item)
                    sequence_cnt = sequence_cnt + 1
                    if sequence_cnt == max_sentence_len - 1:
                        break
            for k in range(sequence_cnt, max_sentence_len):
                X_test[j, k] = np.zeros((1,))
        save_dir = data_dir + data_set + '/train_test_json/' + file + '/'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        print(len(X_train))
        saved_path = save_dir + str(i) + '_x_train.csv'
        X_train = pd.DataFrame(X_train)
        X_train.to_csv(saved_path, index=False, header=True)
        saved_path = save_dir + str(i) + '_y_train.csv'
        Y_train = pd.DataFrame(Y_train)
        Y_train.to_csv(saved_path, index=False, header=True)
        saved_path = save_dir + str(i) + '_x_test.csv'
        X_test = pd.DataFrame(X_test)
        X_test.to_csv(saved_path, index=False, header=True)
        saved_path = save_dir + str(i) + '_y_test.csv'
        Y_test = pd.DataFrame(Y_test)
        Y_test.to_csv(saved_path, index=False, header=True)


def load_train_test(data_dir, data_set, file, index, class_file,
                    embed_size_word2vec=200, encode='utf8'):
    # mybluemix load train test by index
    x_train_file = data_dir + data_set + '/train_test_json/' + \
        file + '/' + str(index) + '_x_train.csv'
    x_train = pd.read_csv(x_train_file, encoding=encode)
    y_train_file = data_dir + data_set + '/train_test_json/' + \
        file + '/' + str(index) + '_y_train.csv'
    y_train = pd.read_csv(y_train_file, encoding=encode)
    x_test_file = data_dir + data_set + '/train_test_json/' + \
        file + '/' + str(index) + '_x_test.csv'
    x_test = pd.read_csv(x_test_file, encoding=encode)
    y_test_file = data_dir + data_set + '/train_test_json/' + \
        file + '/' + str(index) + '_y_test.csv'
    y_test = pd.read_csv(y_test_file, encoding=encode)
    # load any vectors from the word2vec
    embedding_file = data_dir + data_set + '/' + data_set + '.bin'
    print("Load word2vec file {}\n".format(embedding_file))
    word_vectors = KeyedVectors.load_word2vec_format(
        embedding_file, binary=True)
    vocabulary = list(word_vectors.vocab.keys())
    vocabulary.insert(0, 'UNK')
    # initial matrix with random uniform
    embedding = np.random.uniform(
        -0.25,
        0.25,
        (len(word_vectors.vocab), embed_size_word2vec))
    for idx, word in enumerate(word_vectors.vocab):
        embedding[idx] = word_vectors[word]
    lb = LabelBinarizer()
    lb.fit(y_train)
    np.savetxt(class_file, lb.classes_, fmt="%s")
    return x_train.values, y_train.values, x_test.values, y_test.values, embedding, lb


def tokenizer(iterator):
    for value in iterator:
        yield TOKENIZER_RE.findall(value)


def hand(raw_documents, vocabulary):
    for tokens in tokenizer(raw_documents):
        word_ids = [vocabulary.get(token)
                    for token in tokens if vocabulary.get(token) > 0]
        yield word_ids


def pad(documents, document_length):
    for document in documents:
        len_doc = len(document)
        if len_doc < document_length:
            yield np.pad(
                document, (0, document_length - len_doc), 'constant')
        else:
            yield document[: document_length]


def transform_data_hand(
        x_train, y_train,
        x_test, y_test,
        class_file,
        features_names_selected,
        embedding_dim, embedding_file,
        validation=False, x_dev=None, y_dev=None):

    vocabulary = learn.preprocessing.CategoricalVocabulary()
    for feature in features_names_selected:
        vocabulary.add(feature)
    vocabulary.freeze()
    x_train = np.array(list(hand(x_train, vocabulary)))
    # 处理文本数据
    document_length_df = pd.DataFrame([len(xx) for xx in x_train])
    document_length = int(document_length_df.quantile(0.8))

    x_train = pad(x_train, document_length)
    x_test = np.array(list(hand(x_test, vocabulary)))
    x_test = pad(x_test, document_length)
    if validation:
        x_dev = np.array(list(hand(x_dev, vocabulary)))
        x_dev = pad(x_dev, document_length)
    # initial matrix with random uniform
    embedding = np.random.uniform(
        -0.25,
        0.25,
        (len(vocabulary), embedding_dim))
    # load any vectors from the word2vec
    print("Load word2vec file {}\n".format(embedding_file))

    word_vectors = KeyedVectors.load_word2vec_format(
        embedding_file, binary=True)
    for word in word_vectors.vocab:
        idx = vocabulary.get(word)
        if idx != 0:
            embedding[idx] = word_vectors[word]
    lb = LabelBinarizer()
    lb.fit(y_train)
    np.savetxt(class_file, lb.classes_, fmt="%s")
    if validation:
        return x_train, y_train,\
            x_dev, y_dev, x_test, y_test, embedding, lb
    return x_train, y_train, x_test, y_test, document_length, embedding, lb


def transform_data(
        x_train, y_train,
        x_test, y_test,
        class_file,
        features_names_selected,
        embedding_dim, embedding_file,
        validation=False, x_dev=None, y_dev=None):

    # 处理文本数据
    document_length_df = pd.DataFrame(
        [len(xx.split(" ")) for xx in x_train])
    document_length = np.int64(document_length_df.quantile(0.8))

    if features_names_selected:
        vocabulary = learn.preprocessing.CategoricalVocabulary()
        for feature in features_names_selected:
            vocabulary.add(feature)
        vocabulary_processor = \
            learn.preprocessing.VocabularyProcessor(
                document_length, vocabulary=vocabulary)
    else:
        vocabulary_processor = \
            learn.preprocessing.VocabularyProcessor(document_length)

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
    preprocessing_json('../data/bug_triage/', 'chrome', 'classifier_data_101', numCV=3)
    # import pdb
    # pdb.set_trace()
    # a, b, c, d = load_train_test('../data/bug_triage/', 'chrome',
    #                              'classifier_data_101', 7)
