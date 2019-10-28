def tCNN():
    
    # imports
    import os
    import sys
    import numpy as np
    import keras
    from keras.preprocessing.text import Tokenizer
    from keras.utils import to_categorical, plot_model
    from keras.preprocessing.sequence import pad_sequences
    from keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D, ZeroPadding2D
    from keras.layers import MaxPool1D
    from keras.models import Model, Sequential
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adam
    from keras.models import load_model    
 
    visible = Input(shape=(10, 10, 1))
    
    padd1 = ZeroPadding2D(2)(visible)
    padd2 = ZeroPadding2D(2)(visible)
    padd3 = ZeroPadding2D(2)(visible)
    
    conv1 = Conv2D(64, kernel_size=(3, 3), padding='valid', kernel_initializer='he_normal', activation='relu')(padd1)
    conv2 = Conv2D(64, kernel_size=(3, 3), padding='valid', kernel_initializer='he_normal', activation='relu')(padd2)
    conv3 = Conv2D(64, kernel_size=(3, 3), padding='valid', kernel_initializer='he_normal', activation='relu')(padd3)
    
    pool1 = MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid')(conv1)
    pool2 = MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid')(conv2)
    pool3 = MaxPool2D(pool_size=(2, 2), strides=(1,1), padding='valid')(conv3)
    
    pool4 = MaxPool2D(pool_size=(3, 3), strides=(1,1), padding='valid')(pool1)
    pool5 = MaxPool2D(pool_size=(3, 3), strides=(1,1), padding='valid')(pool2)
    pool6 = MaxPool2D(pool_size=(3, 3), strides=(1,1), padding='valid')(pool3)
    
    concatenated_tensor = Concatenate(axis=1)([pool4, pool5, pool6])
    flatten = Flatten()(concatenated_tensor)
    output = Dense(units=6, activation='softmax')(flatten)
    
    model = Model(inputs=visible, outputs=output)

    adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    try:
        model.save('CNN_model_init.h5')
        print("Model Saved!")
    except:
        print("Error in saving Model")