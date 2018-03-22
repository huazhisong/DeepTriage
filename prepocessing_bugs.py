# -*- code:utf-8 -*-

# 导包
import os
import re
import nltk
import pandas as pd
import numpy as np
from dateutil.parser import parse
from datetime import timedelta
from nltk.corpus import stopwords


def select_lines_comments(lines_raw):
    # including summary, description, comments
    # no names
    selected_lines = []
    for line in lines_raw:
        line = line.strip()
        if line == '':
            continue
        if r'[reply] Comment' in line or r'[reply] Description' in line:
            continue
        selected_lines.append(line)
    return selected_lines


def select_lines(lines_raw):
    # select summary and description, no comments
    # no names
    selected_lines = []
    for line in lines_raw:
        line = line.strip()
        if line == '':
            continue
        if r'[reply] Description' in line:
            continue
        if r'[reply] Comment' in line:
            break
        selected_lines.append(line)
    return selected_lines


def select_lines_include_reply(lines_raw):
    # select summary, description, no comments
    # including name
    selected_lines = []
    for line in lines_raw:
        line = line.strip()
        if line == '':
            continue
        if r'[reply] Comment' in line:
            break
        selected_lines.append(line)
    return selected_lines


def clean_raw(raw_text):
    # select summary, description, comments
    # including name
    selected_lines = []
    for line in raw_text:
        line = line.strip()
        if line == '':
            continue
        selected_lines.append(line)
    return selected_lines


# utlize cnn clean_str method to clean
def clean_raw_cnn(raw_text):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", raw_text)
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
    clean_text = re.sub(r"\s{2,}", " ", string)
    return clean_text


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return 'n'


def clean_words(words_raw, wnl, english_stopwords):
    pos_words = nltk.pos_tag(words_raw)
    clean_words = [wnl.lemmatize(word_pos[0], get_wordnet_pos(
        word_pos[1])) for word_pos in pos_words]
    clean_words = ' '.join(clean_words).lower().split()
    clean_words = [
        word for word in clean_words if word not in english_stopwords]
    clean_words = [word for word in clean_words if word.isalpha()]

    return clean_words

# read lines


def read_lines(file_path):
    # open description file
    with open(file_path, encoding='latin2') as f:
        # remove last 5 lines
        lines_raw = f.readlines()
        # read lines specially
        selected_lines = clean_raw(lines_raw)
        # raw text
        raw_text = ' '.join(selected_lines)
        # decode utf8 coding
        raw_text = raw_text.encode('utf8').decode('utf8')
        # sentences tokinzer
        sentences = nltk.sent_tokenize(raw_text)
        tokens = []
        # dealing words
        wnl = nltk.WordNetLemmatizer()
        english_stopwords = stopwords.words('english')
        for sentence in sentences:
            # cean raw sentence
            sentence = clean_raw_cnn(sentence)
            # words tokenizer
            raw_words = nltk.word_tokenize(sentence)
            # clearn word
            tmp = clean_words(raw_words, wnl, english_stopwords)
            tokens.extend(tmp)

        assert len(tokens) > 0
        line = ' '.join(tokens)

    return line


def parse_when(when):
    tz_lookup_tabel = {'EDT': timedelta(hours=12), 'EST': timedelta(
        hours=13), 'PDT': timedelta(hours=15), 'PST': timedelta(hours=16)}
    tz = when.split()[2]
    t = parse(when) + tz_lookup_tabel[tz]
    return t.strftime('%Y-%m-%d %H:%M:%S')


def merged_files(data_files, results_files):
    # 读取文件，处理并写入bugs_all.csv文件
    for bug_file in os.listdir(data_files + '/buglist'):
        bug_dir = bug_file.split('.')[0]
        file_path = data_files + '/buglist/' + bug_file
        with open(file_path, mode='r', encoding='utf8') as f:
            bugs = f.readlines()[1:]
            for bug in bugs:
                tmp = bug.split(',')
                bug_id, bug_who = tmp[0], tmp[4].split(
                    '"')[1] if '"' in tmp[4] else tmp[4]
                bug_assingee = bug_who.split(
                    '@')[0] if '@' in bug_who else bug_who
                print(data_files + '/description/' +
                      bug_dir + '/' + bug_id + '.txt')
                # 读取描述文件
                file_path = data_files +\
                    '/description/' + bug_dir + '/' + bug_id + '.txt'
                line = read_lines(file_path)
                # for raw data
#                 line = "\""+line + "\""
                # 读取修改时间
                print(data_files + '/bughistory_raw/' +
                      bug_dir + '/' + bug_id + '.csv')
                file_path = data_files +\
                    '/bughistory_raw/' + bug_dir + '/' + bug_id + '.csv'
                with open(file_path, encoding='latin2') as f:
                    when = f.readlines()[-1].split(',')[0]
                    when = parse_when(when)
                bug = ','.join([when, bug_id, bug_who, line, bug_assingee])

                # 写入文件
                with open(results_files, mode='a', encoding='utf8') as f:
                    f.write(bug)
                    f.write('\n')


def sortedbytimesplited(results_files):
    train_all = pd.read_csv(results_files, parse_dates=[3])
    train_all_sorted = train_all.sort_values('when')

    # 将数据分成11份
    bug_len = len(train_all_sorted)
    bug_part_size = int((bug_len - 1) / 11) + 1
    for i in range(11):
        begin_index = i * bug_part_size
        end_index = min((i + 1) * bug_part_size, bug_len - 1)
        bug_parted = train_all_sorted.iloc[begin_index:end_index]
        bug_parted.to_csv(data_files + sub_dir + str(i) + '.csv',
                          header=True, index=False)


def get_term_dict(doc_terms_list):
    term_set_dict = {}
    for doc_terms in doc_terms_list:
        for term in doc_terms:
            term_set_dict[term] = 1
    term_set_list = sorted(term_set_dict.keys())  # term set 排序后，按照索引做出字典
    term_set_dict = dict(zip(term_set_list, range(len(term_set_list))))
    return term_set_dict


def get_class_dict(doc_class_list):
    class_set = sorted(list(set(doc_class_list)))
    class_dict = dict(zip(class_set, range(len(class_set))))
    return class_dict


def stats_class_df(doc_class_list, class_dict):
    class_df_list = [0] * len(class_dict)
    for doc_class in doc_class_list:
        class_df_list[class_dict[doc_class]] += 1
    return class_df_list


def stats_term_class_df(doc_terms_list, doc_class_list, term_dict, class_dict):
    term_class_df_mat = np.zeros((len(term_dict), len(class_dict)), np.float32)
    for k in range(len(doc_class_list)):
        class_index = class_dict[doc_class_list[k]]
        doc_terms = doc_terms_list[k]
        for term in set(doc_terms):
            term_index = term_dict[term]
            term_class_df_mat[term_index][class_index] += 1
    return term_class_df_mat


def feature_selection_mi(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    class_set_size = len(class_df_list)

    term_score_mat = np.log(
        ((A + 1.0) * N) / ((A + C) * (A + B + class_set_size)))
    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]

    print(term_set_fs[:10])
    return term_set_fs


def feature_selection_ig(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C = np.tile(class_df_list, (A.shape[0], 1)) - A
    N = sum(class_df_list)
    D = N - A - B - C
    term_df_array = np.sum(A, axis=1)
    class_set_size = len(class_df_list)

    p_t = term_df_array / N
    p_not_t = 1 - p_t
    p_c_t_mat = (A + 1) / (A + B + class_set_size)
    p_c_not_t_mat = (C + 1) / (C + D + class_set_size)
    p_c_t = np.sum(p_c_t_mat * np.log(p_c_t_mat), axis=1)
    p_c_not_t = np.sum(p_c_not_t_mat * np.log(p_c_not_t_mat), axis=1)

    term_score_array = p_t * p_c_t + p_not_t * p_c_not_t
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]

    print(term_set_fs[:10])
    return term_set_fs


def feature_selection_wllr(class_df_list, term_set, term_class_df_mat):
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])
    C_Total = np.tile(class_df_list, (A.shape[0], 1))
    N = sum(class_df_list)
    C_Total_Not = N - C_Total
    term_set_size = len(term_set)

    p_t_c = (A + 1E-6) / (C_Total + 1E-6 * term_set_size)
    p_t_not_c = (B + 1E-6) / (C_Total_Not + 1E-6 * term_set_size)
    term_score_mat = p_t_c * np.log(p_t_c / p_t_not_c)

    term_score_max_list = [max(x) for x in term_score_mat]
    term_score_array = np.array(term_score_max_list)
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    term_set_fs = [term_set[index] for index in sorted_term_score_index]

    print(term_set_fs[:10])
    return term_set_fs


def feature_selection(doc_terms_list, doc_class_list, fs_method, percent):

    class_dict = get_class_dict(doc_class_list)
    term_dict = get_term_dict(doc_terms_list)
    class_df_list = stats_class_df(doc_class_list, class_dict)
    term_class_df_mat = stats_term_class_df(
        doc_terms_list, doc_class_list, term_dict, class_dict)
    term_set = [term[0]
                for term in sorted(term_dict.items(), key=lambda x: x[1])]
    term_set_size = len(term_set)
    selection_size = int(term_set_size * percent)
    term_set_fs = []

    if fs_method == 'MI':
        term_set_fs = feature_selection_mi(
            class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'IG':
        term_set_fs = feature_selection_ig(
            class_df_list, term_set, term_class_df_mat)
    elif fs_method == 'WLLR':
        term_set_fs = feature_selection_wllr(
            class_df_list, term_set, term_class_df_mat)

    return term_set_fs[:selection_size]


def filter_by_feature_selection(data_dir, fs_method='IG',
                                percent=0.7, encoding='utf8',
                                validation=False):
    start_index = 2 if validation else 1
    for step in range(start_index, 11):
        data_files = [data_dir + str(i) + '.csv' for i in range(step + 1)]
        if validation:
            train_data = pd.concat(
                [pd.read_csv(file, encoding=encoding)
                 for file in data_files[:-2]])
        else:
            train_data = pd.concat(
                [pd.read_csv(file, encoding=encoding)
                 for file in data_files[:-1]])

        test_data = pd.read_csv(data_files[-1], encoding=encoding)

        text_train = train_data.text
        fixer_train = train_data.fixer

        text_test = test_data.text
        fixer_test = test_data.fixer
        # 特征选择 待完善
        featurs_names = feature_selection(
            [doc.split() for doc in text_train],
            fixer_train.values, fs_method, percent)

        def filter_by_features_names(doc):
            doc = doc.split()
            tmp = [word for word in doc if word in featurs_names]
            if len(tmp) < 3:
                return doc
            else:
                return ' '.join(tmp)
        print('Training data selection')
        text_train.apply(filter_by_features_names)
        print('Finishing>>>>>>')
        print('Testing data selection')
        text_test.apply(filter_by_features_names)
        print('Finishing>>>>>>')
        data_train = pd.concat([text_train, fixer_train], axis=1)

        results_dir = data_dir + fs_method + str('/')
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        file_train = results_dir + str(percent) + str(step) + '.train.csv'
        data_train.to_csv(file_train, index=False)
        data_test = pd.concat([text_test, fixer_test], axis=1)
        file_test = results_dir + str(percent) + str(step) + '.test.csv'
        data_test.to_csv(file_test, index=False)
        if validation:
            dev_data = pd.read_csv(data_files[-2], encoding=encoding)
            text_dev = dev_data.text
            fixer_dev = dev_data.fixer
            print('Developing data selection')
            text_dev.apply(filter_by_features_names)
            data_dev = pd.concat([text_dev, fixer_dev], axis=1)
            file_dev = results_dir + str(percent) + str(step) + '.dev.csv'
            data_dev.to_csv(file_dev, index=False)
        print('Finishing %s' % step)


if __name__ == '__main__':
    """
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    """
    # 文件存储位置
    data_dir = '../../../data/'
    data_set = '/eclipse'
    data_files = data_dir + data_set
    sub_dir = '/song_no_select/'
    results_files = data_files + sub_dir + 'bugs_all.csv'
    # 写入列名
    with open(results_files, 'w') as f:
        f.write('when,id,who,text,fixer\n')
    # 合并文件
    merged_files(data_files, results_files)
    # 将文件分成十一份
    sortedbytimesplited(results_files)
