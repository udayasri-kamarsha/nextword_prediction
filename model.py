import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import pickle

# Load text from a file
with open('C:/Users/siva sathivada/Desktop/miniproject/next/SherlockHolmes.txt', 'r', encoding='utf-8') as file:
    text_data = file.read()

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_data])

# Save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

input_sequences = []
for sentence in text_data.split('\n'):
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1, len(tokenized_sentence)):
        input_sequences.append(tokenized_sentence[:i + 1])

max_len = max(len(x) for x in input_sequences)
padded_input_sequences = pad_sequences(input_sequences, maxlen=max_len, padding='pre')

x = padded_input_sequences[:, :-1]
y = padded_input_sequences[:, -1]
y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)


model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 100, input_length=max_len - 1))
model.add(LSTM(150))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x, y, epochs=100, verbose=1)

model.save('next_word_model.h5')
