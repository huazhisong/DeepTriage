#!/usr/lib/env python3
# -*- code: latin-1 -*-

import numpy as np
import pandas as pd
import csv

data_dir = "../../data/data_by_ocean/Eclipse_raw/"
bug_list_dir = "buglist/"
bug_description_dir = "description/"
bug_history_dir = "bughistory_raw/"
with open(data_dir + 'time_raw.csv', 'a', encoding='latin-1') as f:
    f.write('when,bug_id,summary,description,who')
for i in range(20):
    j = i + 1
    bug_list_record = pd.read_csv(data_dir + bug_list_dir + 'bugs' + str(j) + '.csv', encoding='latin-1')
    for bug in bug_list_record.values:
        print(bug)
        print(len(bug))
        print('description: '+data_dir + bug_description_dir + 'bugs' + str(i + 1) + '/' + str(bug[0]) + '.txt')
        with open(data_dir + bug_description_dir + 'bugs' + str(i + 1) + '/' + str(bug[0]) + '.txt', 'r',
                  encoding='latin-1') as description_f:
            bug_description_record = " ".join(map(str.strip, description_f.readlines()))
        with open(data_dir + bug_history_dir + 'bugs' + str(i + 1) + '/' + str(bug[0]) + '.csv', 'r', encoding='latin-1') as history_f:
            data = [line.strip().split(',')[:5] for line in history_f.readlines()]
            with open(data_dir + bug_history_dir + 'bugs' + str(i + 1) + '/' + str(bug[0]) + '.delt.csv', 'w', encoding='latin-1') as write_f:
                for line in data:
                    write_f.write(",".join(line)+'\n')
        print('history: ' + data_dir + bug_history_dir + 'bugs' + str(i + 1) + '/' + str(bug[0]) + '.delt.csv')
        bug_history_record = pd.read_csv(data_dir + bug_history_dir + 'bugs' + str(i + 1) + '/' + str(bug[0]) + '.delt.csv',quoting=csv.QUOTE_NONE, encoding='latin-1')
        bug_history = bug_history_record.sort_values(['When'], ascending=False)
        bug_history_record = bug_history[(bug_history[' Added'] == 'FIXED') & (bug_history[' What'] == 'Resolution')]
        if len(bug_history_record.values):
            with open(data_dir + 'time_raw.csv', 'a', encoding='latin-1') as f:
                line = bug_history_record['When'].values[0] + ',' + str(bug[0]) + ',' + str(
                    bug[8]) + ',' + bug_description_record + ',' + \
                       bug_history_record[' Who'].values[0]
                # f.write(line + '\n')
