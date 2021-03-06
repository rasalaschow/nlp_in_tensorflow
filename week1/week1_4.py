# Sarcasm Detection using Hybrid Neural Network
# Rishabh Misra, Prahal Arora
# Arxiv, August 2019

# Params:
# @ is_sarcastic: 1 if it is sarcastic otherwise 0
# @ headline: the headline of the news article
# @ article_link: link to the original news article

import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequence

# !wget --no-check-certificate \
#     https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \
#     -O /tmp/sarcasm.json


with open("/archive/sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequence(sentences)
padded = pad_sequence(seqences, padding='post')
