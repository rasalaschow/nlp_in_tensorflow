import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.proprocessing.sequence import pad_sequence
# Corpus
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my cat!',
    'Do you think my dog is amazing'
]

tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequence(sequences, maxlen=5)
print("\n Word Index: ", word_index)
print("\n Sequences: ", sequences)
print("\n Padded Sequence:")
print(padded)

test_corpus = [
    'I really love my dog',
    'my dog loves my manatee'
]

# In this case, some words are not in the original word dict
# so the output is only including the existing indexes
test_seq = tokenizer.texts_to_sequences(test_corpus)
print("\nTest Sequences", test_seq)
padded = pad_sequence(test_seq, maxlen=10)
print("\n Padded Test Sequences:")
print(padded)
