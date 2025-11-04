import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
from keras import Sequential
from keras.layers import SimpleRNN
import tensorflow as tf

# model = Sequential([SimpleRNN(4, (3, 2))])
x = tf.random.normal((1, 3, 2))

# layer = SimpleRNN(4, input_shape=(3, 2))
layer = SimpleRNN(4, input_shape=(3, 2), return_sequences=True)
output = layer(x)

print(output.shape)