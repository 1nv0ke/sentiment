"""
Author: Chang Sun
Date:   Dec 5, 2017
Description:
    This module contains various functions for creating a table mapping tweets and the user.
"""

import json

# _________________________________________________________________________________________________

TEXT_KEY_LENGTH = None

DELIMITER = '-tweet number#:'

URL_USER_ID = 'https://twitter.com/intent/user?user_id='
URL_USER_NAME = 'https://twitter.com/'
URL_TWEET_ID = 'https://twitter.com/statuses/'

# _________________________________________________________________________________________________

def parse_tweet_json(json_file, text_file, output_file):
    map_text_ids = dict()
    cnt_pool = 0
    with open(json_file) as f_json:
        for line in f_json.read().decode('utf-8').splitlines():
            cnt_pool += 1
            data = json.loads(line)
            text = data['text'][:TEXT_KEY_LENGTH]
            map_text_ids[text] = (data['id_str'], data['user']['id_str'])
    print('Processed %d tweets. Created dictionary of %d tweets.' % (cnt_pool, len(map_text_ids)))

    cnt_tweets = 0
    with open(text_file) as f_text, open(output_file, 'w') as f_out:
        text = ''
        for line in f_text.read().decode('utf-8').splitlines()[1:]:
            if line.find(DELIMITER) != -1:
                if text:
                    text = text[:-1]
                    if text[:TEXT_KEY_LENGTH] in map_text_ids:
                        tweet_id_str, user_id_str = map_text_ids[text[:TEXT_KEY_LENGTH]]
                        f_out.write(','.join((str(cnt_tweets), URL_TWEET_ID + tweet_id_str, URL_USER_ID + user_id_str)) + '\n')
                    else:
                        print('Warning: cannot find tweet #%d in dictionary.' % (cnt_tweets))
                    text = ''
                    cnt_tweets += 1
            else:
                text += line + '\n'
        if text:
            text = text[:-1]
            if text[:TEXT_KEY_LENGTH] in map_text_ids:
                tweet_id_str, user_id_str = map_text_ids[text[:TEXT_KEY_LENGTH]]
                f_out.write(','.join((str(cnt_tweets), URL_TWEET_ID + tweet_id_str, URL_USER_ID + user_id_str)) + '\n')
            else:
                print('Warning: cannot find tweet #%d in dictionary.' % (cnt_tweets))
            cnt_tweets += 1
    print('Created tweet ID and user ID mapping for %d tweets.' % (cnt_tweets))

# _________________________________________________________________________________________________

if __name__ == '__main__':
    parse_tweet_json('./data/raw/tw_with_id.json',
                     './data/text/tweets.txt',
                     './data/raw/tweets_url.csv')

# _________________________________________________________________________________________________
