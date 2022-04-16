import math

import numpy as np
import pandas as pd
from sklearn import preprocessing
import datetime as dt
import talib
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Conv1D, Conv2D, MaxPooling1D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import RepeatVector, TimeDistributed
import os
import glob
import re
import tensorflow.keras as keras
import sklearn
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import TensorBoard
from time import sleep
from keras.applications.resnet import ResNet50
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
np.random.seed(42)

def BinClassModel2():
    model = Sequential()
    model.add(Bidirectional(LSTM(80, return_sequences=True), input_shape=(40, 40)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(80, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(80, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model

def BinClassModel():
    model = Sequential()
    model.add(Bidirectional(LSTM(800, return_sequences=True), input_shape=(40, 40)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(800, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(800, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model