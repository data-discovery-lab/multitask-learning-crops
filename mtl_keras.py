from keras.callbacks import Callback
from keras.engine.input_layer import Input
from keras.engine.training import Model
from keras.layers.core import Dense
from pandas.core.frame import DataFrame
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from keras import backend as K
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing.data import MinMaxScaler

df = pd.read_csv('data/data_excel_converted.csv', delimiter=',')

# ensure all data is float
# values = df.values.astype('float32')
# normalize features
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled = scaler.fit_transform(values)
# df = DataFrame(scaled)

samples = 475
features = 42
tasks = 3
test_size = int(0.2*samples)

### shape = (samples, features)
dat = []
for i in range(50):
    if i < 8:
        continue
    dat.append(df[df.columns[i]])

dat = np.array(dat).transpose()

# Generate indexes of test and train
idx_list = np.linspace(0, samples-1, num=samples)
idx_test = np.random.choice(samples, size=test_size, replace=False)
idx_train = np.delete(idx_list, idx_test).astype('int')

# Split data into test and train
dat_train = dat[idx_train, :]
labels = df[df.columns[3:6]]

label_train_1 = labels.values[idx_train, 0] # year 2000
label_train_2 = labels.values[idx_train, 1] # year 2001
label_train_3 = labels.values[idx_train, 2] # year 2002

dat_test = dat[idx_test, :]
label_test_1 = labels.values[idx_test, 0] # year 2000
label_test_2 = labels.values[idx_test, 1] # year 2000
label_test_3 = labels.values[idx_test, 2] # year 2000

x = Input(shape=(features, ))
shared = Dense(units=4*features)(x)

sub1 = Dense(units=16, activation='relu')(shared)
sub2 = Dense(units=16, activation='relu')(shared)
sub3 = Dense(units=16, activation='relu')(shared)

out1 = Dense(units=1, activation='linear')(sub1)
out2 = Dense(units=1, activation='linear')(sub2)
out3 = Dense(units=1, activation='linear')(sub3)


model = Model(inputs=x, outputs=[out1, out2, out3])

def correlation_coefficient_loss(y_true, y_pred):
    # x = y_true
    # y = y_pred
    # mx = K.mean(x)
    # my = K.mean(y)
    # xm, ym = x-mx, y-my
    # r_num = K.sum(tf.multiply(xm,ym))
    # r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    # r = r_num / r_den
    #
    # r = K.maximum(K.minimum(r, 1.0), -1.0)
    # return 1 - K.square(r)
    print("y=", y_true)
    return mean_squared_error(y_true, y_pred)
# def my_loss_function(a,b):
#     return mean_squared_log_error(a, b)

# Compiling the model using 'adam' optimizer and MSE as loss function
model.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mse', 'mae', 'mape'],  loss_weights=[1.0, 1.0, 1.0])

#muti_outputs shape= tasks x train_samples
callbacks = []
model.fit(x=dat_train, y=[label_train_1, label_train_2, label_train_3], epochs=5000, batch_size=20)

pred1, pred2, pred3 = model.predict(dat_test)

# inv_y = scaler.inverse_transform(inv_y)

plot_x = dat_test[:, 0]
plot_y = pred1.flatten() - label_test_1
plt.scatter(x=plot_x, y=plot_y)

print("test mse=", mean_squared_error(label_test_1, pred1.flatten()))
print("me=", max(abs(plot_y)))
plt.show()
