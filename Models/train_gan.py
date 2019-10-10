import sys
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers

K.set_image_dim_ordering('th')

if(len(sys.argv) != 3):
        print("usage : python3 train_gan.py digit choice \n where choice = simple or rich or EMNIST1 or EMNIST2")
else:
        if(int(sys.argv[1]) >= 0 and int(sys.argv[1]) < 10): 
                digit = int(sys.argv[1])
        else: 
                digit = 8
        if(sys.argv[2] == "rich" or sys.argv[2] == "EMNIST1" or sys.argv[2] == "EMNIST2"): 
                choice = sys.argv[2] 
        else: 
                choice = "simple"

        if(choice == "EMNIST1"):
                train_images = np.load('emnist1.npy')
                train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

        elif(choice == "EMNIST2"):
                train_images = np.load('emnist2.npy')
                train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

        else:
                (x_train, y_train), (_, _) = mnist.load_data() 

                if(choice == "rich"):
                        x_augmented = []
                        y_augmented = []

                        for i in range(x_train.shape[0]):
                                # original image
                                x_augmented.append(x_train[i])
                                y_augmented.append(y_train[i])

                                # rotated image
                                x_augmented.append((tf.contrib.keras.preprocessing.image.random_rotation(x_train[i].reshape(28,28,1), 20, row_axis=0, col_axis=1, channel_axis=2)).reshape(28,28))
                                y_augmented.append(y_train[i])

                                # random shear
                                x_augmented.append((tf.contrib.keras.preprocessing.image.random_shear(x_train[i].reshape(28,28,1), 20, row_axis=0, col_axis=1, channel_axis=2)).reshape(28,28))
                                y_augmented.append(y_train[i])

                                # random shift
                                x_augmented.append((tf.contrib.keras.preprocessing.image.random_shift(x_train[i].reshape(28,28,1), 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2)).reshape(28,28))
                                y_augmented.append(y_train[i])
        
                        x_train = np.array(x_augmented)
                        y_train = np.array(y_augmented)

                temp = []
                counter = 0
                for i in range(x_train.shape[0]):
                        if (y_train[i] == digit):
                                counter += 1
                                temp.append(x_train[i].tolist())

                train_images = np.array(temp)
                train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')

        X_train = train_images
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
        X_train = X_train.reshape(X_train.shape[0], 784)

        # Constructing the two networks
        adam = Adam(lr=0.0002, beta_1=0.5)
        randomDim = 10

        generator = Sequential()
        generator.add(Dense(256, input_dim=randomDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(512))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(1024))
        generator.add(LeakyReLU(0.2))
        generator.add(Dense(784, activation='tanh'))
        generator.compile(loss='binary_crossentropy', optimizer=adam)

        discriminator = Sequential()
        discriminator.add(Dense(1024, input_dim=784, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(512))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(256))
        discriminator.add(LeakyReLU(0.2))
        discriminator.add(Dropout(0.3))
        discriminator.add(Dense(1, activation='sigmoid'))
        discriminator.compile(loss='binary_crossentropy', optimizer=adam)

        # Construct the combined network
        discriminator.trainable = False
        ganInput = Input(shape=(randomDim,))
        x = generator(ganInput)
        ganOutput = discriminator(x)
        gan = Model(inputs=ganInput, outputs=ganOutput)
        gan.compile(loss='binary_crossentropy', optimizer = adam)

        epochs = 1000 
        batchSize = 128
        train_size = X_train.shape[0]

        batchCount = train_size / batchSize

        for _ in range(1,epochs+1):
                for _ in tqdm(range(int(batchCount))):

                        # Get a batch of MNIST images
                        imageBatch = X_train[np.random.randint(0, train_size, size=batchSize)]

                        # Generate a batch of images using the generator
                        noise = np.random.normal(0, 1, size=[batchSize, randomDim])
                        generatedImages = generator.predict(noise)

                        # Train discriminator
                        # First, create the batch: MNIST images get label 0.9, generated images get label 0
                        X = np.concatenate([imageBatch, generatedImages])
                        yDis = np.zeros(2*batchSize)
                        yDis[:batchSize] = 0.9
			# Then, train
                        discriminator.trainable = True
                        discriminator.train_on_batch(X, yDis)

                        # Train generator
                        noise = np.random.normal(0, 1, size=[batchSize, randomDim])
                        yGen = np.ones(batchSize)
                        discriminator.trainable = False
                        gan.train_on_batch(noise, yGen)

        generator.save('gan_digit%d_%s.h5' % (digit,choice))


