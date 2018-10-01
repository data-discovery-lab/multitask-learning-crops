from keras.callbacks import Callback
from keras.engine.input_layer import Input
from keras.engine.training import Model
from keras.layers.core import Dense
from pandas.core.frame import DataFrame
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error
from keras import backend as K
import tensorflow as tf
from sklearn.utils import check_array

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing.data import MinMaxScaler

from keras.regularizers import l2

def kernel_regu(weight_matrix:None):
    print("done")

    return 0


def activity_regu(output:None):
    print('done!!!!!!!!!!!!!!!!!!!!', output)

    return 0


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
grid_cells = None
grid_cells_id = None
for i in range(50):
    if i == 0:
        grid_cells = df[df.columns[i]].values
    elif i == 1:
        grid_cells_id = df[df.columns[i]].values
    if i < 8:
        continue
    dat.append(df[df.columns[i]])

dat = np.array(dat).transpose()

# Generate indexes of test and train
idx_list = np.linspace(0, samples-1, num=samples)
idx_test = np.random.choice(samples, size=test_size, replace=False)
idx_train = np.delete(idx_list, idx_test).astype('int')

grid_cells_test = grid_cells[idx_test]
grid_cells_id_test = grid_cells[idx_test]

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

out1 = Dense(units=1, activation='linear', kernel_regularizer=kernel_regu, activity_regularizer=activity_regu )(sub1)
out2 = Dense(units=1, activation='linear')(sub2)
out3 = Dense(units=1, activation='linear')(sub3)


model = Model(inputs=x, outputs=[out1, out2, out3])


def mean_absolute_percentage_error(y_true, y_pred):
    # y_true = check_array(y_true)
    # y_pred = check_array(y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true):
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def my_loss_function(y_true, y_pred):

    mean_square_loss = K.mean(K.square(y_pred - y_true), axis=-1)
    neighbor_size = 1
    rng = np.random.RandomState(42)
    ## 1 neighbor
    neighbors = rng.randn(test_size, neighbor_size).astype('float32')

    mean_average_neighbor_loss = K.mean(K.square(y_pred - K.sum(neighbors) / neighbor_size), axis=-1)

    return mean_square_loss + mean_average_neighbor_loss

    # return mean_squared_error(y_true, y_pred)

# Compiling the model using 'adam' optimizer and MSE as loss function
model.compile(optimizer='adam', loss=my_loss_function,  metrics=['mse', 'mae', 'mape'],  loss_weights=[1.0, 1.0, 1.0])
# model.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mse', 'mae', 'mape'],  loss_weights=[1.0, 1.0, 1.0])

#muti_outputs shape= tasks x train_samples
callbacks = []
model.fit(x=dat_train, y=[label_train_1, label_train_2, label_train_3], epochs=5, batch_size=20)

pred1, pred2, pred3 = model.predict(dat_test)

# inv_y = scaler.inverse_transform(inv_y)

plot_x = dat_test[:, 0]
plot_y = pred1.flatten() - label_test_1
# plt.scatter(x=plot_x, y=plot_y)

final_data = np.array([
    grid_cells_test,
    label_test_1,
    pred1.flatten(),
    grid_cells_id_test
]).transpose()

## sort by first column
final_data = final_data[final_data[:, 0].argsort()]

final_x = final_data[:, 0]
final_label = final_data[:, 1]
final_prediction = final_data[:, 2]
final_cell_id = final_data[:, 3]
# final_data.sort(axis=1)

my_output = pd.DataFrame({'fid': final_x, 'id': final_cell_id, 'label': final_label, 'predicted': final_prediction})
my_output.to_csv('output/prediction.csv', index=False)

plt.plot(final_x,  final_prediction, 'red')
plt.plot(final_x, final_label, 'blue')

print("test mse=", mean_squared_error(label_test_1, pred1.flatten()))
print("test mape=", mean_absolute_percentage_error(label_test_1, pred1.flatten()))
print("test mae=", mean_absolute_error(label_test_1, pred1.flatten()))
print("me=", max(abs(plot_y)))
plt.show()
