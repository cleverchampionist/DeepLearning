from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.datasets import imdb

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = 2000)

# Process the data
x_train = sequence.pad_sequences(x_train, maxlen = 80)
x_test = sequence.pad_sequences(x_test, maxlen = 80)

# Create the model
model = Sequential()
model.add(Embedding(2000, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#Train the model
model.fit(x_train, y_train, batch_size = 32, epochs = 15, verbose = 1, validation_data=(x_test, y_test))

#Evaluate the model
score, acc = model.evaluate(x_test, y_test, batch_size=32)
print('Test score', score)
print('Test accuracy:', acc)
