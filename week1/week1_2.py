from tensorflow.keras.preprocessing import Tokenizer
# add padding
from tensorflow.keras.preprocessing.sequence import pad_sequence

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my cat!',
    'Do you think my dog is amazing'
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

# make all the sequence into the same length and be placed at last
# the matrix width is the same as the longest one
padded = pad_sequence(sequences)
print(padded)

# default pre is from beginning
# add the parameter padding equals post, the index will be placed at first place
# add the parameter truncating equals post, the index will be lost from the end
padded2 = pad_sequence(sequences, padding='post', truncating='post', maxlen=5)

test_data = [
    'I really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)
