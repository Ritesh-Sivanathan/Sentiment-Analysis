
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

from pp import preprocess_text

# Just some data ops

dataset = pd.read_csv('Tweets.csv')

dataset['clean_text'] = dataset['text'].apply(preprocess_text) # create new column called clean_text in dataset, applies preprocess_text func to preproecss data in the text column and store it in the clean_text column. preprocessory.py

tokenizer = Tokenizer()
tokenizer.fit_on_texts(dataset['clean_text']) # tokenizes words to sequences of integers
sequences = tokenizer.texts_to_sequences(dataset['clean_text'])

max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# ---------------------------------------------------------------------------------------------

model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=max_length),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

label_encoder = LabelEncoder()
dataset['airline_sentiment'] = label_encoder.fit_transform(dataset['airline_sentiment'])

history = model.fit(padded_sequences, dataset['airline_sentiment'], epochs=10, validation_split=0.2)

test_dataset = pd.read_csv('Tweets.csv')

test_dataset['clean_text'] = test_dataset['text'].apply(preprocess_text)

test_sequences = tokenizer.texts_to_sequences(test_dataset['clean_text'])
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_length, padding='post')

dataset['airline_sentiment'] = label_encoder.fit_transform(dataset['airline_sentiment'])

loss, accuracy = model.evaluate(test_padded_sequences, dataset['airline_sentiment'])
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

# model.save('kewlmodel.h5')
