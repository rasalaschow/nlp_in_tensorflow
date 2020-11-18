import tensorflow as tf
print(tf.__versoin__)
tf.enable_eager_execution()

# if you are using colab
!pip install -q tensorflow-datasets

import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

train_data, test_data = imdb['train'], imdb['test']

train_sent = []
train_labels = []

test_sent = []
test_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
# they are originally tensor!!
for s, l in train_data:
    train_sent.append(str(s.numpy()))
    train_labels.append(l.numpy())

for s, l in test_data:
    test_sent.append(str(s.numpy()))
    test_labels.append(l.numpy())

# example
tf.Tensor(b"As a lifelong fan of Dickens, I have invariably been disappointed by adaptations of his novels.<br />", shape=(), dtype=string)

tr.Tensor(1, shape=(), dtype=int64)

train_labels_final = np.array(train_labels)
test_labels_final = np.array(test_labels)

# tokenize
vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'

tokenizer= Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sent)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequence(train_sent)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

test_seqences = tokenizer.texts_to_sequence(test_sent)
test_padded = pad_sequences(test_seqences, maxlen=max_length)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
