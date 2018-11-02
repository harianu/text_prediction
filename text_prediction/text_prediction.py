
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as kerasutils
import numpy as np

tokens = Tokenizer()


def cleansing(data):
    text_data = data.lower().split("\n")
    tokens.fit_on_texts(text_data)
    total_words = len(tokens.word_index) + 1
    input_sequences = []
    for line in text_data:
        token_list = tokens.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[:i+1]
            input_sequences.append(n_gram_sequence)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(
        input_sequences, maxlen=max_sequence_len, padding='pre'))
    predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
    label = kerasutils.to_categorical(label, num_classes=total_words)
    return predictors, label, max_sequence_len, total_words


def build_model(predictors, label, max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(LSTM(150, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    earlystop = EarlyStopping(
        monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model.fit(predictors, label, epochs=100, verbose=1, callbacks=[earlystop])
    print(model.summary())
    return model


def next_words(seed_text, next_words, max_sequence_len):
    for _ in range(next_words):
        token_list = tokens.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences(
            [token_list], maxlen=max_sequence_len-1, padding='pre')
        prediction = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokens.word_index.items():
            if index == prediction:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text


data = open('data.txt').read()

predictors, label, max_sequence_len, total_words = cleansing(data)
model = build_model(predictors, label, max_sequence_len, total_words)
print(next_words("minister", 2, max_sequence_len))
