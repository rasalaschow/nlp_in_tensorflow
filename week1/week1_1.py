import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

# Corpus
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my cat!'
]

# num_words means it only takes first 100 words of the sentence
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)

# get the index of the word and they are in dict format.
word_index = tokenizer.word_index
print(word_index)

# Text to Sequence, in list format
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

test_corpus = [
    'I really love my dog',
    'my dog loves my manatee'
]

# In this case, some words are not in the original word dict
# so the output is only including the existing indexes
test_seq = tokenizer.texts_to_sequences(test_corpus)
print(test_seq)
