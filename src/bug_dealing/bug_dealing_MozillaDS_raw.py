#!/usr/lib/env python3
# -*- code: latin-1 -*-

# import
import os
import pandas as pd
import csv
from dateutil import parser
from pandas.tseries.offsets import Hour

# 数据位置
data_dir = "../../data/data_by_ocean/MozillaDS_raw/"
bug_list_dir = "buglist/"
bug_description_dir = "description_raw/"
bug_history_dir = "bughistory_raw/"

# 如果没有处理，则读入源数据，进行处理
if not os.path.exists(data_dir + 'raw/bug_raw.csv'):
    # 将所有数据都放入bug_raw.csv文件，分割符为@@,,@@，很无奈，处理的文本包括太多的字符了
    with open(data_dir + 'raw/bug_raw.csv', 'a', encoding='latin-1') as f:
        f.write('when@@,,@@bug_id@@,,@@summary@@,,@@description@@,,@@who\n')
    # 分别处理20个文件里面的bug
    for i in range(22):
        j = i + 1
        # 读取 bug list 的文件，获取bug id和summary
        bug_list_record = pd.read_csv(
            data_dir + bug_list_dir +
            'bugs' + str(j) + '.cgi', encoding='latin-1')
        # 针对bug list里面的每一个bug，我们将读取其描述文件和bug history文件
        for bug in bug_list_record.values:
            print(bug)
            print(len(bug))
            print('description: ' + data_dir + bug_description_dir +
                  'bugs' + str(i + 1) + '/' + str(bug[0]) + '.txt')
            # 读取bug的description文件，简单的将每行文本提取出来，并去掉多余空格，组合起来便是一个描述性文件
            with open(data_dir + bug_description_dir +
                      'bugs' + str(i + 1) + '/' + str(bug[0]) + '.txt', 'r',
                      encoding='latin-1') as description_f:
                bug_description_record = " ".join(
                    map(str.strip, description_f.readlines()))
            # 预处理history文件，原始文件格式不统一，使用pd.read_csv方法总是失败
            # 故此处只处理前五列的数据，以方便下文处理
            if not os.path.exists(data_dir + bug_history_dir +
                                  'bugs' + str(i + 1) + '/' +
                                  str(bug[0]) + '.delt.csv'):
                with open(data_dir + bug_history_dir + 'bugs' +
                          str(i + 1) + '/' + str(bug[0]) + '.csv', 'r',
                          encoding='latin-1') as history_f:
                    data = [line.strip().split(',')[:5]
                            for line in history_f.readlines()]
                    with open(data_dir + bug_history_dir + 'bugs' +
                              str(i + 1) + '/' +
                              str(bug[0]) + '.delt.csv', 'w',
                              encoding='latin-1') as write_f:
                        for line in data:
                            write_f.write(",".join(line) + '\n')

            print('history: ' + data_dir + bug_history_dir + 'bugs' +
                  str(i + 1) + '/' + str(bug[0]) + '.delt.csv')
            # 读取history文件
            bug_history_record = pd.read_csv(
                data_dir + bug_history_dir + 'bugs' +
                str(i + 1) + '/' + str(bug[0]) + '.delt.csv',
                quoting=csv.QUOTE_NONE, encoding='latin-1')
            # 将history按时间排序，提取 Added和What数据
            bug_history = bug_history_record.sort_values(
                ['When'], ascending=False)
            try:
                bug_history_record = \
                    bug_history[(bug_history[' Added'] ==
                                 'FIXED') &
                                (bug_history[' What'] == 'Resolution')]
            except KeyError:
                bug_history_record = \
                    bug_history[(bug_history['Added'] ==
                                 'FIXED') &
                                (bug_history['What'] == 'Resolution')]
            # 如果成功提取到history数据，则将问问分别
            # 写入bug_summary_raw,bug_description_raw,bug_list_raw,bug_raw文件
            # 其中bug_raw文件是汇总文件
            if len(bug_history_record.values):
                with open(data_dir + 'raw/bug_summary_raw.csv',
                          'a', encoding='latin-1') as f:
                    f.write(str(bug[8]) + '\n')
                with open(data_dir + 'raw/bug_description_raw.csv',
                          'a', encoding='latin-1') as f:
                    f.write(bug_description_record + '\n')
                with open(data_dir + 'raw/bug_list_raw.csv',
                          'a', encoding='latin-1') as f:
                    try:
                        f.write(
                            bug_history_record['When'].values[0] + ',' +
                            bug_history_record[' Who'].values[0] + '\n')
                    except KeyError:
                        f.write(
                            bug_history_record['When'].values[0] + ',' +
                            bug_history_record['Who'].values[0] + '\n')
                with open(data_dir + 'raw/bug_raw.csv',
                          'a', encoding='latin-1') as f:
                    try:
                        line = bug_history_record['When'].values[0] +\
                            '@@,,@@' + str(bug[0]) + '@@,,@@' + str(
                            bug[8]) + '@@,,@@' + \
                            bug_description_record + '@@,,@@' + \
                            bug_history_record[' Who'].values[0]
                        f.write(line + '\n')
                    except KeyError:
                        line = bug_history_record['When'].values[0] +\
                            '@@,,@@' + str(bug[0]) + '@@,,@@' + str(
                            bug[8]) + '@@,,@@' + \
                            bug_description_record + '@@,,@@' + \
                            bug_history_record['Who'].values[0]
                        f.write(line + '\n')
else:
    def date_parse(time):
        tz = time.split(' ')[2]
        dt = parser.parse(time)
        if tz == 'EDT':
            return dt + Hour(12)
        elif tz == 'EST':
            return dt + Hour(13)
        elif tz == 'PDT':
            return dt + Hour(15)
        elif tz == 'PST':
            return dt + Hour(16)
        else:
            print('缺少时区：%s' % tz)

    # 读取预处理文件
    bug_raw = pd.read_csv(data_dir + 'raw/bug_raw.csv',
                          sep='@@,,@@',
                          engine='python',
                          encoding='latin-1',
                          parse_dates=[0],
                          date_parser=date_parse)
    # 按照时间将数据进行排序
    bug_sorted_raw = bug_raw.sort_values('when')
    # 将排序好的数据写入磁盘备份
    bug_sorted_raw.to_csv(data_dir + 'raw/sorted_summary_description.csv',
                          columns=['description', 'summary'],
                          header=False, index=False)
    bug_sorted_raw.to_csv(data_dir + 'raw/sorted_bug_id_date_who.csv',
                          columns=['when', 'bug_id', 'who'], index=False)
    # 将数据分成11份
    bug_len = len(bug_sorted_raw)
    bug_part_size = int((bug_len - 1) / 11) + 1
    for i in range(11):
        begin_index = i * bug_part_size
        end_index = min((i + 1) * bug_part_size, bug_len - 1)
        bug_parted = bug_sorted_raw.iloc[begin_index:end_index]
        bug_parted.to_csv(data_dir + str(i) + '.csv',
                          header=False, index=False)
