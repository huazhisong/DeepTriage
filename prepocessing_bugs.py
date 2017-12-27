# -*- code:utf-8 -*-

# 导包
import os
import re
import nltk
import pandas as pd
from dateutil.parser import parse
from datetime import timedelta
from nltk.corpus import stopwords


# 文件存储位置
data_dir = '../../../data'
data_set = '/eclipse'
data_files = data_dir + data_set
sub_dir = '/song_no_select_summary_description/'
results_files = data_files + sub_dir + 'bugs_all.csv'


# select summary, description and comments
def select_lines_include_reply(lines_raw):
    selected_lines = []
    for line in lines_raw:
        line = line.strip()
        if line == '':
            continue
        if r'[reply] Comment' in line:
            break
        selected_lines.append(line)
    return selected_lines


def select_lines_comments(lines_raw):
    selected_lines = []
    for line in lines_raw:
        line = line.strip()
        if line == '':
            continue
        if r'[reply] Comment' in line or r'[reply] Description' in line:
            continue
        selected_lines.append(line)
    return selected_lines

# select summary and description, no comments


def select_lines(lines_raw):
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

# do nothing but strip() methed


def clean_raw(raw_text):
    return raw_text.strip()


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
        word for word in clean_words if not word in english_stopwords]
    clean_words = [word for word in clean_words if word.isalpha()]

    return clean_words

# read lines


def read_lines(file_path):
    # open description file
    with open(file_path, encoding='latin2') as f:
        # remove last 5 lines
        lines_raw = f.readlines()[:-5]
        # read lines specially
        selected_lines = select_lines_include_reply(lines_raw)
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
        line = ' '.join(tokens)

    return line


def parse_when(when):
    tz_lookup_tabel = {'EDT': timedelta(hours=12), 'EST': timedelta(
        hours=13), 'PDT': timedelta(hours=15), 'PST': timedelta(hours=16)}
    tz = when.split()[2]
    t = parse(when) + tz_lookup_tabel[tz]
    return t.strftime('%Y-%m-%d %H:%M:%S')


def merged_files():
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
                file_path = data_files + '/description/' + bug_dir + '/' + bug_id + '.txt'
                line = read_lines(file_path)
                # for raw data
#                 line = "\""+line + "\""
                # 读取修改时间
                print(data_files + '/bughistory_raw/' +
                      bug_dir + '/' + bug_id + '.csv')
                file_path = data_files + '/bughistory_raw/' + bug_dir + '/' + bug_id + '.csv'
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


if __name__ == '__main__':
    # 写入列名
    with open(results_files, 'w') as f:
        f.write('when,id,who,text,fixer\n')
    # 合并文件
    merged_files()
    # 将文件分成十一份
    sortedbytimesplited(results_files)
