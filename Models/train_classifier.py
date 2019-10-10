import sys 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np

if(len(sys.argv) != 3):
        print("usage : python3 train_classifier.py capacity augmented \n where capacity = 1 or 2 or 3 and augmented = simple or rich")
else:

        if(int(sys.argv[1]) >= 1 and int(sys.argv[1]) < 4): 
                capacity = int(sys.argv[1])
        else: 
                capacity = 1
        if(sys.argv[2] == "rich"): 
                augmented = "rich"
        else: 
                augmented = "simple"

        if(capacity == 1):
                size = [32,64,200]
        elif(capacity == 2):
                size = [64,128,256]
        else:
                size = [64,128,512]

        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        train_size = x_train.shape[0]
        test_size = x_test.shape[0]

        x_train = x_train.reshape(train_size,28,28,1)
        x_test = x_test.reshape(test_size,28,28,1)
        input_shape = (28,28,1)

        x_train = x_train.astype('float32')
        x_train /= 255
        x_test = x_test.astype('float32')
        x_test /= 255
        
        if(augmented == "rich"):
                x_augmented = []
                y_augmented = []

                for i in range(train_size):
                        # original image
                        x_augmented.append(x_train[i])
                        y_augmented.append(y_train[i])

                        # rotated image
                        x_augmented.append(tf.contrib.keras.preprocessing.image.random_rotation(x_train[i].reshape(28,28,1), 20, row_axis=0, col_axis=1, channel_axis=2))
                        y_augmented.append(y_train[i])

                        # random shear
                        x_augmented.append(tf.contrib.keras.preprocessing.image.random_shear(x_train[i].reshape(28,28,1), 20, row_axis=0, col_axis=1, channel_axis=2))
                        y_augmented.append(y_train[i])

                        # random shift
                        x_augmented.append(tf.contrib.keras.preprocessing.image.random_shift(x_train[i].reshape(28,28,1), 0.2, 0.2, row_axis=0, col_axis=1, channel_axis=2))
                        y_augmented.append(y_train[i])

                x_train = np.array(x_augmented)
                y_train = np.array(y_augmented)
                              
        # Using Tensorflow backend
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Activation, Dense, Conv2D, Dropout, Flatten, MaxPooling2D

        # Creating a sequential model and adding the layers
        model = tf.keras.Sequential()

        model.add(Conv2D(size[0],kernel_size=(3,3),input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(size[0], (3, 3)))
        model.add(Activation('relu'))

        model.add(Conv2D(size[1], (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(size[1], (3, 3)))
        model.add(Activation('relu'))

        model.add(Flatten()) #Flattening the 2D arrays for fully connected layers
        model.add(Dense(size[2],activation=tf.nn.relu))
        model.add(Dense(size[2],activation=tf.nn.relu))
        model.add(Dropout(0.1))
        model.add(Dense(10,activation=tf.nn.softmax))

        # Compiling the model
        model.compile(optimizer='adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])
        # Fitting the model
        model.fit(x_train,y_train,epochs=3)
        # Evaluate the model
        print(model.evaluate(x_test, y_test))
        # Save the model
        model.save('classifier_capacity%d_%s.model' % (capacity,augmented))
        tf.keras.experimental.export_saved_model(model, 'classifier')


