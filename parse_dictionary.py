"""
Author: Chang Sun
Date:   Dec 5, 2017
Description:
    This module contains functions for generating the dictionary.
"""

import re
import json
import emoji
from collections import Counter

# _________________________________________________________________________________________________

DELIMITER = '-tweet number#:'

# _________________________________________________________________________________________________

def parse_tweet_json(json_file, words_dict_file, emojis_dict_file, max_word_count=None, max_emoji_count=None):
    all_words = Counter()
    all_emojis = Counter()
    cnt_tweets = 0

    with open(json_file) as f_json:
        for line in f_json.read().decode('utf-8').splitlines():
            cnt_tweets += 1
            data = json.loads(line)
            text = data['text']
            words = re.compile('\w+').findall(text)
            for word in words:
                if len(word) > 1:
                    all_words[word.lower()] += 1
            for c in emoji.UNICODE_EMOJI:
                if c in text:
                    all_emojis[c] += text.count(c)
    print('Processed %d tweets. Found %d unique letter words and %d emojis.' % (cnt_tweets, len(all_words), len(all_emojis)))

    with open(words_dict_file, 'w') as f_out:
        word_cnt_pairs = sorted(all_words.items(), key=lambda pair: pair[1], reverse=True)
        for pair in word_cnt_pairs[:max_word_count]:
            f_out.write(pair[0].encode('utf-8') + '\n')
    print('Saved %d words into dictionary.' % (min(max_word_count, len(all_words))))

    with open(emojis_dict_file, 'w') as f_out:
        emoji_cnt_pairs = sorted(all_emojis.items(), key=lambda pair: pair[1], reverse=True)
        for pair in emoji_cnt_pairs[:max_emoji_count]:
            f_out.write(pair[0].encode('utf-8') + '\n')
    print('Saved %d emojis into dictionary.' % (min(max_emoji_count, len(all_emojis))))

# _________________________________________________________________________________________________

if __name__ == '__main__':
    parse_tweet_json('./data/raw/tw_with_id.json',
                     './data/dict/words.dict',
                     './data/dict/emojis.dict',
                     max_word_count=4000, max_emoji_count=1000)

# _________________________________________________________________________________________________
