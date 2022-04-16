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
dat1 = glob.glob("E:/BiPOSTrainingData/1/*")[0:10000]
#dat1_ngoaiMy = glob.glob("E:/BiPOSTemplingData/1/*")
#dat1 = dat1_ngoaiMy + dat1_ArgBrz
dat0 = glob.glob("E:/BiPOSTrainingData/0/*")
np.random.shuffle(dat0)
dat0=dat0[0:30000:3]
print(len(dat1))
print(len(dat0))
imbalanced = len(dat1)/(len(dat0)+len(dat1))
print("Độ lệch dữ liệu: ", imbalanced, 1- imbalanced)
#sleep(100)

# Chia train test
dat0_train, dat0_test = train_test_split(dat0, test_size=0.1, random_state=0)
dat1_train, dat1_test = train_test_split(dat1, test_size=0.1, random_state=0)
# Chia train validation
dat0_train, dat0_val = train_test_split(dat0_train, test_size=0.2, random_state=0)
dat1_train, dat1_val = train_test_split(dat1_train, test_size=0.2, random_state=0)

train_file = dat0_train + dat1_train
test_file = dat0_test + dat1_test
val_file = dat0_val + dat1_val
np.random.shuffle(train_file)
np.random.shuffle(val_file)
np.random.shuffle(test_file)
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
                pattern = tf.constant(eval("file[21:22]"))
                for j in range(len(label_class)):
                    if re.match(pattern.numpy(), label_class[j].numpy()):
                        #k = tf.keras.utils.to_categorical(j, num_classes=2)
                        nhãn.append(j)
    # Nếu dùng onehot như trên đây thì đầu ra là vector nên phải dùng softmax
    # Không dùng mã hóa Onehot thì có thể xuất đầu ra sigmoid và Dense1
            dữ_liệu = np.array(dữ_liệu).reshape(-1,40,40)
            nhãn = np.array(nhãn)
            yield dữ_liệu, nhãn
            i = i+1
            #print(nhãn)

batch_size = 20
train_set = tf.data.Dataset.from_generator(bộ_phát_dữ_liệu, args=[train_file, batch_size], output_types=(tf.float32, tf.float32))
val_set = tf.data.Dataset.from_generator(bộ_phát_dữ_liệu, args=[val_file, batch_size], output_types=(tf.float32, tf.float32))
test_set = tf.data.Dataset.from_generator(bộ_phát_dữ_liệu, args=[test_file, batch_size], output_types=(tf.float32, tf.float32))

# Kiểm tra generate data
num = 0
for data, label, calssw in train_set:
    print(data.shape, label.shape)
    print("Label bắt đầu: ", label)
    print(calssw)
    #print(np.argmax(label, axis=None))
    num += 1
    if num>4: break



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
full_model.load_weights('New_1.47-0.1413-0.9529-0.1644.hdf5')

# Thiết lập Optimizer
lrate_start = 0.001
lrate = keras.optimizers.schedules.ExponentialDecay(lrate_start, decay_steps=60000, decay_rate=0.75)
opti = keras.optimizers.Adam(learning_rate=lrate_start)

# Thiết lập metrics
loss = 'binary_crossentropy'
metrics = [BinaryAccuracy(name='acc'),
           TruePositives(name='TP'),
           TrueNegatives(name='TN'),
           FalsePositives(name='FP'),
           FalseNegatives(name='FN')] #sử dụng với no onehot và dense1 sigmoid
full_model.compile(loss=loss, optimizer=opti, metrics=metrics)#[tf.keras.metrics.Accuracy()]

steps_per_epoch = int(np.ceil(len(train_file)/batch_size))
val_steps = int(np.ceil(len(val_file)/batch_size))
test_steps = int(np.ceil(len(test_file)/batch_size))
#print("Số bước trên epoch:", steps_per_epoch)
print("Số bước validation: ", val_steps)
print("Số bước test: ", test_steps)

filepath="New_2.{epoch:02d}-{loss:.4f}-{acc:.4f}-{val_loss:.4f}.hdf5"
call = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_weights_only=True, save_best_only=True, mode='auto', save_freq='epoch')
history=full_model.fit(train_set, validation_data=val_set, steps_per_epoch=steps_per_epoch,batch_size=batch_size,
                       validation_steps=val_steps, epochs=50, callbacks=[call])
#full_model.load_weights('ArgBrz1_2_800.20-0.2066-0.9182-0.2028.hdf5')
#full_model.evaluate(test_set)
plot_history(history, yrange=(0.9, 1))
#full_model.save('Bidirect3ClassModel.h5')

