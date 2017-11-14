
import re
from collections import Counter

# _________________________________________________________________________________________________

DICT_DIR = './data/dict/'
WORDS_DICTIONARY_FILE = 'words.dict'
EMOJIS_DICTIONARY_FILE = 'emojis.dict'

TEXT_DIR = './data/text/'
TEXT_FILE = 'tweets.txt'
DATA_FILE = 'tweets.data'

DELIMITER = '-tweet number#:'

# _________________________________________________________________________________________________

DIR_LABEL = './data/label/'
NAMES = ['chang', 'hao', 'yecheng', 'yue']
CONSENSUS_FILE = 'consensus.label'

# _________________________________________________________________________________________________

def parse_raw_labels(dir, raw_label_file, parsed_label_file):
    '''
    Parse raw label files (containing empty lines, etc.)
    '''
    labels = []

    with open(dir + raw_label_file) as f:
        labels = [int(x) for x in f.read().splitlines() if x]
    print('Found %d labels in file "%s"' % (len(labels), raw_label_file))

    with open(dir + parsed_label_file, 'w') as f:
        for label in labels:
            f.write(str(label) + '\n')
    print('Parsed labels saved in file %s' % (parsed_label_file))


def consensus_rule(labels):
    '''
    Consensus rule for determining the ground-truth labels
    '''
    return max(set(labels), key=labels.count)


def create_consensus_labels(parse_raw=False):
    label_files = [x + '.label' for x in NAMES]

    if parse_raw:
        raw_files = [x + '.raw_label' for x in NAMES]
        for i in range(len(label_files)):
            parse_raw_labels(DIR_LABEL, raw_files[i], label_files[i])

    all_labels = []

    for file in label_files:
        labels = []
        with open(DIR + file) as f:
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

def parse_tweets():
    dict_words = dict()
    word_cnt = 0
    for file in [WORDS_DICTIONARY_FILE, EMOJIS_DICTIONARY_FILE]:
        curr_cnt = word_cnt
        with open(DICT_DIR + file) as f:
            for word in f.read().decode('utf-8').splitlines():
                word_cnt += 1
                dict_words[word] = word_cnt
        print('Added %d words from %s to dictionary' % (word_cnt - curr_cnt, file))
    print('Dictionary word count: %d' % (word_cnt))

    with open(TEXT_DIR + TEXT_FILE) as f_in, open(TEXT_DIR + DATA_FILE, 'w') as f_out:
        tweet_cnt = 0
        word_counter = Counter()
        for line in f_in.read().decode('utf-8').splitlines()[1:461]:
            line = line.strip().lower()
            print line
            print tweet_cnt
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

if __name__ == '__main__':
    pass

# _________________________________________________________________________________________________
