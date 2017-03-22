from __future__ import print_function
import numpy as np
import keras
#from keras.utils import to_categorical
from keras.utils import np_utils
from keras.datasets import mnist, cifar10
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
import keras.regularizers


batch_size = 128
num_classes = 10
#epochs = 5

#mnist image dimensionality
img_rows = 32
img_cols = 32

#loading the mnist dataInit
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

divided_input = np.array_split(X_train, 50)
X_train = divided_input[0]


divided_output = np.array_split(Y_train, 50)
Y_train = divided_output[0]

divided_inputtest = np.array_split(X_test, 50)
X_test = divided_inputtest[0]
divided_outputtest = np.array_split(Y_test, 50)
Y_test = divided_outputtest[0]


#reshaping for input to network
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
input_shape = (img_rows, img_cols, 3)

#making data float datatype
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#normalizing the data
X_train /= 255
X_test /= 255

print('x_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#convert class vectors to one hot encoded vectors
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)


feature_layers = [
    Convolution2D(64, 5, 5, border_mode='same',subsample=(2, 2), input_shape=(32, 32, 3)),
    LeakyReLU(0.2),
    Dropout(0.5),
    Convolution2D(128, 5, 5, border_mode='same', subsample=(2,2)),
    LeakyReLU(0.2),
    Dropout(0.5),
    Convolution2D(256, 5, 5, border_mode='same', subsample=(2,2)),
    LeakyReLU(0.2),
    Dropout(0.5),
    Convolution2D(512, 5, 5, border_mode='same', subsample=(4,4)),
    LeakyReLU(0.2),
    Dropout(0.5),
    Flatten()
]

classification_layers = [
    Dense(512, W_regularizer=keras.regularizers.l2(0.01), name='fc_layer1'),
    Activation('relu'),
    Dense(num_classes, activation='softmax', W_regularizer=keras.regularizers.l2(0.01), name='fc_layer2')
]


model = Sequential(feature_layers + classification_layers)
# different backend has different image dim order, so we need to judge first.
'''
input_shape = (28,28,1)
model.add(Convolution2D(64, 5, 5, border_mode='same',subsample=(2, 2), input_shape=input_shape))
#model.add(LeakyReLU(0.02))
model.add(Activation('tanh'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(128, 5, 5, border_mode='same', subsample=(2,2)))
#model.add(LeakyReLU(0.02))
#model.add(BatchNormalization())
model.add(Activation('tanh'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1024))
#model.add(LeakyReLU(0.02))
#model.add(BatchNormalization())
model.add(Activation('tanh'))
model.add(Dense(num_classes, activation='softmax'))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))
'''
#print model.summary()

model.load_weights('discriminator_cifar', by_name=True)

for l in feature_layers:
    l.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

print('Model Compilation successful')



model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=100,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
