from __future__ import print_function
import numpy as np
import keras
#from keras.utils import to_categorical
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras import regularizers
#from keras.regularizers import
#from keras.utils import plot_model

#hyperparameters of the model
batch_size = 128
num_classes = 10
epochs = 200

#mnist image dimensionality
img_rows = 28
img_cols = 28

#loading the mnist dataInit
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)

def get_data(label, n):
    #X = np.empty(n, img_rows, img_cols, 1)
    #Y = np.empty(n, 1)
    X = []
    Y = []
    #for i in range(num_classes):
    index = np.where(Y_train == label)[0]
    print(len(index))
    #np.append(X, X_train[index,:,:,:])
    X = X_train[index,:]
    #X = X[n,:]
    Y = Y_train[index]
    #Y = Y[n]
    #return X, Y
    X = np.asarray(X)
    X = X[:n]
    print('X', X.shape)
    Y = np.asarray(Y)
    Y = Y[:n]
    return X, Y
#print(X[2].shape, Y.shape)
X_10, Y_10 = get_data(0, 100)
#print(X_10.shape, Y_10.shape)
#image = X_10[2,:,:]
#image = image.reshape((28,28))
#print(image.shape)
#plt.imshow(image, cmap='gray')
#plt.show()
X_1, Y_1 = get_data(1, 100)
X_2, Y_2 = get_data(2, 100)
X_3, Y_3 = get_data(3, 100)
X_4, Y_4 = get_data(4, 100)
X_5, Y_5 = get_data(5, 100)
X_6, Y_6 = get_data(6, 100)
X_7, Y_7 = get_data(7, 100)
X_8, Y_8 = get_data(8, 100)
X_9, Y_9 = get_data(9, 100)
X = np.concatenate((X_10, X_5, X_2, X_8, X_1, X_4, X_9, X_3, X_6, X_7))
Y = np.concatenate((Y_10, Y_5, Y_2, Y_8, Y_1, Y_4, Y_9, Y_3, Y_6, Y_7))
print(X.shape, Y.shape)
permutation = np.random.permutation(X.shape[0])
X_train = X[permutation]
Y_train = Y[permutation]
'''
divided_input = np.array_split(X_train, 300)
X_train = divided_input[0]
divided_output = np.array_split(Y_train, 300)
Y_train = divided_output[0]

print (Y_train.shape)

divided_inputtest = np.array_split(X_test, 100)
X_test = divided_inputtest[0]
divided_outputtest = np.array_split(Y_test, 100)
Y_test = divided_outputtest[0]
'''

#reshaping for input to network
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

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
print (Y_train.shape)


#reshaping for input to network
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

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
    Convolution2D(64, 5, 5, border_mode='same',subsample=(2, 2), W_regularizer=regularizers.l2(0.01), input_shape=input_shape),
    #BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Convolution2D(128, 5, 5, border_mode='same', subsample=(2,2), W_regularizer=regularizers.l2(0.01)),
    #BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Flatten(),
    Dense(1024),
    #BatchNormalization(),
    Activation('relu'),
    Dropout(0.5)
    #Dense(1, name='dense')
]

classification_layers = [
    #Dense(512, name='fc_layer1'),
    #BatchNormalization(),
    #Activation('relu'),
    #Dropout(0.7),
    Dense(num_classes, activation='softmax', W_regularizer=regularizers.l2(0.01), name='fc_layer2')
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

model.load_weights('discriminator', by_name=True)

for l in feature_layers:
    l.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

print('Model Compilation successful')



model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
#plot_model(model, to_file='model.png')
