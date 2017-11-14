"""
Author: Chang Sun
Date:   Nov 13, 2017
Description:
    This module contains various functions for creating the twitter sentiment analysis dataset.
"""

import re
import random
from collections import Counter

# _________________________________________________________________________________________________

DIR_DICT = './data/dict/'
WORDS_DICTIONARY_FILE = 'words.dict'
EMOJIS_DICTIONARY_FILE = 'emojis.dict'

DIR_TEXT = './data/text/'
TEXT_FILE = 'tweets.txt'
DATA_FILE = 'tweets.data'

DELIMITER = '-tweet number#:'

DIR_DATA = './data/'
TRAINING_DATA_FILE = 'train.data'
TRAINING_LABEL_FILE = 'train.label'
TEST_DATA_FILE = 'test.data'
TEST_LABEL_FILE = 'test.label'

# _________________________________________________________________________________________________

DIR_LABEL = './data/label/'
NAMES = ['chang', 'hao', 'yecheng', 'yue']
CONSENSUS_FILE = 'consensus.label'

# _________________________________________________________________________________________________

"""
Functions for parsing and generating labels
"""

def parse_raw_labels(dir, raw_label_file, parsed_label_file):
    """
    Parse raw label files (containing empty lines, etc.)
    """
    labels = []

    with open(dir + raw_label_file) as f:
        labels = [int(x) for x in f.read().splitlines() if x]
    print('Found %d labels in file "%s"' % (len(labels), raw_label_file))

    with open(dir + parsed_label_file, 'w') as f:
        for label in labels:
            f.write(str(label) + '\n')
    print('Parsed labels saved in file %s' % (parsed_label_file))


def consensus_rule(labels):
    """
    Consensus rule for determining the ground-truth labels
    """
    return max(set(labels), key=labels.count)


def create_consensus_labels(parse_raw=False):
    """
    Generate consensus labels from various labelers
    """
    label_files = [x + '.label' for x in NAMES]

    if parse_raw:
        raw_files = [x + '.raw_label' for x in NAMES]
        for i in range(len(label_files)):
            parse_raw_labels(DIR_LABEL, raw_files[i], label_files[i])

    all_labels = []

    for file in label_files:
        labels = []
        with open(DIR_LABEL + file) as f:
            labels = [int(x) for x in f.read().splitlines()]
        print('Loaded %d labels from %s' % (len(labels), file))
        all_labels.append(labels)

    # Transpose
    all_labels = map(list, zip(*all_labels))
    joint_labels = [consensus_rule(labels) for labels in all_labels]

    with open(DIR_LABEL + CONSENSUS_FILE, 'w') as f:
        for label in joint_labels:
            f.write(str(label) + '\n')

    print('Saved %d consensus labels to %s' % (len(joint_labels), CONSENSUS_FILE))

# _________________________________________________________________________________________________

"""
Functions for parsing tweets text file
"""

def parse_tweets():
    """
    Parse tweets text file
    """
    dict_words = dict()
    word_cnt = 0
    for file in [WORDS_DICTIONARY_FILE, EMOJIS_DICTIONARY_FILE]:
        curr_cnt = word_cnt
        with open(DIR_DICT + file) as f:
            for word in f.read().decode('utf-8').splitlines():
                word_cnt += 1
                dict_words[word] = word_cnt
        print('Added %d words from %s to dictionary' % (word_cnt - curr_cnt, file))
    print('Dictionary word count: %d' % (word_cnt))

    with open(DIR_TEXT + TEXT_FILE) as f_in, open(DIR_TEXT + DATA_FILE, 'w') as f_out:
        tweet_cnt = 0
        word_counter = Counter()
        for line in f_in.read().decode('utf-8').splitlines()[3:463]:
            line = line.strip().lower()
            if line.find(DELIMITER) != -1:
                tweet_cnt += 1
                if word_counter:
                    for key in sorted(word_counter.keys()):
                        f_out.write('%d %d %d\n' % (tweet_cnt, key, word_counter[key]))
                    word_counter = Counter()
            else:
                for word in list(line) + re.compile('\w+').findall(line):
                    if word in dict_words:
                        word_counter[dict_words[word]] += 1
        print('Successfully parsed %d tweets' % (tweet_cnt))

# _________________________________________________________________________________________________

def split_dataset(folds=5):
    """
    Splits the dataset into training set and test set
    """
    with open(DIR_TEXT + DATA_FILE) as f_all_data, \
         open(DIR_LABEL + CONSENSUS_FILE) as f_label, \
         open(DIR_DATA + TRAINING_DATA_FILE, 'w') as f_train_data, \
         open(DIR_DATA + TRAINING_LABEL_FILE, 'w') as f_train_label, \
         open(DIR_DATA + TEST_DATA_FILE, 'w') as f_test_data, \
         open(DIR_DATA + TEST_LABEL_FILE, 'w') as f_test_label:

        labels = f_label.readlines()
        label_cnt = len(labels)
        test_cnt = label_cnt / folds
        train_cnt = label_cnt - test_cnt
        testset = sorted(random.sample(range(1, label_cnt+1), test_cnt))

        # Create label files
        mapped_train = dict()
        mapped_test = dict()
        train_num = 0
        test_num = 0
        for i in range(label_cnt):
            if i+1 in testset:
                test_num += 1
                mapped_test[i+1] = test_num
                f_test_label.write(labels[i])
            else:
                train_num += 1
                mapped_train[i+1] = train_num
                f_train_label.write(labels[i])

        # Create input data files
        data_lines = f_all_data.readlines()
        for i in range(len(data_lines)):
            tokens = data_lines[i].split()
            curr_num = int(tokens[0])
            remaining = ' ' + ' '.join(tokens[1:]) + '\n'
            if curr_num in testset:
                f_test_data.write(str(mapped_test[curr_num]) + remaining)
            else:
                f_train_data.write(str(mapped_train[curr_num]) + remaining)

        print('Splitted dataset into %d for training and %d for testing' % (train_cnt, test_cnt))

# _________________________________________________________________________________________________

if __name__ == '__main__':
    create_consensus_labels()
    parse_tweets()
    split_dataset(folds=5)

# _________________________________________________________________________________________________
