import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, Cropping2D,BatchNormalization
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
from utils import INPUT_SHAPE, batch_generator, load_func
from pathlib import Path
np.random.seed(0)
def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_path = Path('driving_log.csv')
    try:
        data_df = load_func(data_path)
    except FileNotFoundError:
        print(f"File '{data_path}' not found.")
        return None
    
    # Handle missing or invalid data if needed
    data_df.dropna(inplace=True)

    X = data_df[['forward', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid

def build_model(args):
    """
    Improved NVIDIA model with batch normalization, dropout, and more fully connected layers
    Model with best output
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dropout(args.keep_prob))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(args.keep_prob))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(args.keep_prob))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.summary()
    return model

#def build_model(args):
"""
    Simpler Model without labda and less connected layer
"""
#    model = Sequential()
#    model.add(Conv2D(16, (3, 3), input_shape=(160, 320, 3), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Conv2D(32, (3, 3), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Conv2D(64, (3, 3), activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Flatten())
#    model.add(Dense(500, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(100, activation='relu'))
#    return model

#def build_model(args):
"""
Improved NVIDIA model with batch normalization, cropping and regularizer
"""
#    model = Sequential()
#    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
#    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
#    # Data augmentation
#    model.add(RandomFlip(mode='horizontal'))
#    model.add(RandomRotation(factor=0.1))
#    model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
#    model.add(BatchNormalization())
#    model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
#    model.add(BatchNormalization())
#    model.add(Conv2D(48, (5, 5), activation='relu', strides=(2, 2)))
#    model.add(BatchNormalization())
#    model.add(Conv2D(64, (3, 3), activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Conv2D(64, (3, 3), activation='relu'))
#    model.add(BatchNormalization())
#    model.add(Flatten())
#    # Regularization with L2 weight decay and Dropout
#    l2_reg = 0.001  # Adjust the regularization strength
#    dropout_rate = 0.2  # Adjust the dropout rate
#    model.add(Dense(1164, activation='relu', kernel_regularizer=l2(l2_reg)))
#    model.add(Dropout(dropout_rate))
#    model.add(Dense(100, activation='relu', kernel_regularizer=l2(l2_reg)))
#    model.add(Dropout(dropout_rate))
#    model.add(Dense(50, activation='relu', kernel_regularizer=l2(l2_reg)))
#    model.add(Dropout(dropout_rate))
#    model.add(Dense(10, activation='relu', kernel_regularizer=l2(l2_reg)))
#    model.add(Dropout(dropout_rate)) 
#    model.add(Dense(1))
#    model.summary()    
#    return model

def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('model-v2-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=args.learning_rate))
    model.fit(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
          steps_per_epoch=args.samples_per_epoch,
          epochs=args.nb_epoch,
          validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
          validation_steps=len(X_valid),
          callbacks=[checkpoint],
          verbose=1)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-a', help='data directory',        dest='l2_reg',          type=float,   default=0.001)
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()

