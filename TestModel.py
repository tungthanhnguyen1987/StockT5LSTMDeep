import math
from BinaryModel import BinClassModel2
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
from keras.metrics import BinaryAccuracy, TrueNegatives, TruePositives, FalseNegatives, FalsePositives, binary_accuracy
from sklearn.utils import class_weight
np.random.seed(42)
#dat1 = glob.glob("E:/TestingData/1/*")
#dat0 = glob.glob("E:/TestingData/0/*")
dat1 = glob.glob("E:/BiPOSTrainingData/1/*")[0:31000]
dat0 = glob.glob("E:/BiPOSTrainingData/0/*")[0:31000]
#np.random.shuffle(dat1)
#np.random.shuffle(dat0)
n=10090
test_file = dat0[n:n+5] + dat1[n:n+5]
print("Showing test file...")
print(test_file)
#sleep(100)
def bộ_phát_dữ_liệu (list_file, batch_size):
    i=0
    while True:
        if i*batch_size >= len(list_file): # Chạy generator vô hạn
            i = 0
            np.random.shuffle(list_file)
        else:
            nhóm_file = list_file[i*batch_size:(i+1)*batch_size]
            dữ_liệu = []
            nhãn = []
            label_class = tf.constant(["0", "1"])
            sumne = 0
            sumpo = 0
            for file in nhóm_file:
                temp = pd.read_csv(open(file, 'r'))
                dữ_liệu.append(temp)
                #dữ_liệu= np.concatenate()([dữ_liệu, dữ_liệu, dữ_liệu])
                #print(file)
                #sleep(1000)
                pattern = tf.constant(eval("file[15:16]"))
                for j in range(len(label_class)):
                    if re.match(pattern.numpy(), label_class[j].numpy()):
                        #k = tf.keras.utils.to_categorical(j, num_classes=2)
                        nhãn.append(j)
    # Nếu dùng onehot như trên đây thì đầu ra là vector nên phải dùng softmax
    # Không dùng mã hóa Onehot thì có thể xuất đầu ra sigmoid và Dense1
            dữ_liệu = np.array(dữ_liệu).reshape(-1,40,40)
            nhãn = np.array(nhãn)
            yield dữ_liệu #,nhãn
            i = i+1
            #print(nhãn)

batch_size = 1
test_set = tf.data.Dataset.from_generator(bộ_phát_dữ_liệu, args=[test_file, batch_size], output_types=(tf.float32))
"""
#Kiểm tra generate data
num = 0
for data, label in test_set:
    print(data.shape, label.shape)
    print("Label bắt đầu: ", label)
    #print(calssw)
    #print(np.argmax(label, axis=None))
    num += 1
    if num>4: break
"""


def plot_history (history, yrange):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc)
    plt.plot(epochs, val_acc)
    plt.title('Training and validation accuracy')
    plt.ylim(yrange)
    plt.figure()
    plt.plot(epochs, loss)
    plt.plot(epochs, val_loss)
    plt.title('Training and validation loss')
    plt.show()

full_model = BinClassModel2()

#full_model.load_weights('ArgBrz1_10_800.08-0.0680-0.9779-0.2672.hdf5')

# Thiết lập Optimizer
lrate_start = 0.000005
lrate = keras.optimizers.schedules.ExponentialDecay(lrate_start, decay_steps=60000, decay_rate=0.75)
opti = keras.optimizers.Adam(learning_rate=lrate_start)

# Thiết lập metrics
loss = 'binary_crossentropy'
metrics = [BinaryAccuracy(name='acc'),
           TruePositives(name='TP'),
           TrueNegatives(name='TN'),
           FalsePositives(name='FP'),
           FalseNegatives(name='FN')] #sử dụng với no onehot và dense1 sigmoid
#classw = {0: 0.2, 1: 0.8}

full_model.compile(loss=loss, optimizer=opti, metrics=metrics)#[tf.keras.metrics.Accuracy()]
#full_model.save('ArgentinaBrazil.h5')
#steps_per_epoch = int(np.ceil(len(train_file)/batch_size))
#val_steps = int(np.ceil(len(val_file)/batch_size))
steps = int(np.ceil(len(test_file)/batch_size))
#print("Số bước trên epoch:", steps_per_epoch)
#print("Số bước validation: ", val_steps)
print("Số bước test: ", steps)

#filepath="Val04ArgBrz1_1_800.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}.hdf5"
#call = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True, mode='auto', save_freq='epoch')
full_model.load_weights('New_1.24-0.1810-0.9390-0.1635.hdf5')
pred=full_model.predict(test_set, verbose=1, steps=steps)
res=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
for i in range(len(pred)):
    if pred[i]>=0.5:
        res[i]=1
    elif pred[i]<0.5:
        res[i]=0
print(pred)
print(res[:10])
#history=full_model.fit(train_set, validation_data=val_set, steps_per_epoch=steps_per_epoch,batch_size=batch_size,
 #                      validation_steps=val_steps, epochs=100, callbacks=[call])


#plot_history(history, yrange=(0.9, 1))
