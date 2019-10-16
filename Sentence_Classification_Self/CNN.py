def tCNN():
    
    # imports
    import os
    import sys
    import numpy as np
    import keras
    from keras.preprocessing.text import Tokenizer
    from keras.utils import to_categorical, plot_model
    from keras.preprocessing.sequence import pad_sequences
    from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D
    from keras.layers import MaxPool1D
    from keras.models import Model, Sequential
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adam
        
 

    model = Sequential()
    
    model.add(Conv2D(128, kernel_size=(3, 3), padding='valid', kernel_initializer='normal', activation='relu', input_shape = (10, 10, 3)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid'))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='valid', kernel_initializer='normal', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid'))
    model.add(Conv2D(32, kernel_size=(3, 3), padding='valid', kernel_initializer='normal', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid'))
    
    model.add(Flatten())
    model.add(Dense(units=8, activation='softmax'))

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()