from keras import backend as K
from keras.layers import Layer
from keras.models import Sequential
from keras.layers import Dense

class MyCustomLayer(Layer):
    
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyCustomLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                    shape=(input_shape[1], self.output_dim), initializer='normal', trainable=True)
        super(MyCustomLayer, self).build(input_shape)
        
    def call(self, input_data):
        return K.dot(input_data, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],  self.output_dim)

model = Sequential()
model.add(MyCustomLayer(32, input_shape=(16,)))
model.add(Dense(8, activation='softmax'))
model.summary()
