import numpy as np
import matplotlib 
import matplotlib.pyplot as plot
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical

model = Sequential()
model.add(Dense(512, input_shape = (784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

#Hidden layer
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
# print('model is now ready-to-use')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print("Loaded data")

# for i in range(10):
#     plot.subplot(3, 5, i+1)
#     plot.tight_layout()
#     plot.imshow(x_train[i], cmap='gray', interpolation='none')
#     plot.title("Digit: {}".format(y_train[i]))
#     plot.xticks([])
#     plot.yticks([])
# plot.show()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

# plot.hist(x_train[0])
# plot.title("Digit: {}".format(y_train[0]))
# plot.show()

# print (np.unique(y_train, return_counts=True))
n_classes = 10
y_train = to_categorical(y_train, n_classes)
y_test = to_categorical(y_test, n_classes)
# for i in range(5):
#     print(y_train[i])

history = model.fit(x_train, y_train, batch_size = 128,
            epochs = 10, verbose = 2, validation_data = (x_test, y_test))
loss_and_metrics = model.evaluate(x_test, y_test, verbose = 2)

print("Test Loss:", loss_and_metrics[0])
print("Test Accuracy:", loss_and_metrics[1])

plot.subplot(2, 1, 1)
plot.plot(history.history['accuracy'])
plot.plot(history.history['val_accuracy'])
plot.title('model accuracy')
plot.ylabel('accuracy')
plot.xlabel('epoch')
plot.legend(['train', 'test'], loc = 'lower right')

plot.subplot(2, 1, 2)
plot.plot(history.history['loss'])
plot.plot(history.history['val_loss'])
plot.title('model loss')
plot.ylabel('loss')
plot.xlabel('epoch')
plot.legend(['train', 'test'], loc = 'upper right')
plot.tight_layout()
plot.show()

predictions = model.predict(x_test)
corret_predictions = np.nonzero(predictions == y_test)[0]
incorrect_predictions = np.nonzero(predictions != y_test)[0]

print(len(corret_predictions), "classified correctly")
print(len(incorrect_predictions), "classified incorrectly")


