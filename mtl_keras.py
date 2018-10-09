from keras.callbacks import Callback
from keras.engine.input_layer import Input
from keras.engine.training import Model
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from pandas.core.frame import DataFrame
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error
from keras import backend as K
import tensorflow as tf
from keras.optimizers import SGD
from sklearn.utils import check_array

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing.data import MinMaxScaler

from keras.regularizers import l2

# def kernel_regu(weight_matrix:None):
#     print("done")
#
#     return 0
#
#
# def activity_regu(output:None):
#     print('done!!!!!!!!!!!!!!!!!!!!', output)
#
#     return 0


df = pd.read_csv('data/data_excel_converted.csv', delimiter=',')
df.dropna(inplace=True)

# df['prediction'] = np.array(np.random.randn(len(df.index)))

# df.to_csv('output/prediction_2003.csv', index=False)# ensure all data is float
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
np.random.seed(6)
idx_test = np.random.choice(samples, size=test_size, replace=False)
idx_train = np.delete(idx_list, idx_test).astype('int')

grid_cells_test = grid_cells[idx_test]
grid_cells_id_test = grid_cells_id[idx_test]

# Split data into test and train
dat_train = dat[idx_train, :]
labels = df[df.columns[3:7]]

label_train_1 = labels.values[idx_train, 3] # year 2003
label_train_2 = labels.values[idx_train, 2] # year 2002
label_train_3 = labels.values[idx_train, 1] # year 2001
label_train_4 = labels.values[idx_train, 0] # year 2000

dat_test = dat[idx_test, :]
label_test_1 = labels.values[idx_test, 3] # year 2003
label_test_2 = labels.values[idx_test, 2] # year 2002
label_test_3 = labels.values[idx_test, 1] # year 2001
label_test_4 = labels.values[idx_test, 0] # year 2000

x = Input(shape=(features, ))
shared = Dense(units=4*features)(x)

sub1 = Dense(units=64, activation='relu')(shared)
sub2 = Dense(units=64, activation='relu')(shared)
sub3 = Dense(units=64, activation='relu')(shared)
#
# # lstm1 = LSTM(16, return_sequences=True)(sub1)
# d1 = Dense(units=1, activation='relu')(sub1)
# drop1 = Dropout(0.1)(d1)
# d2 = Dense(units=1, activation='relu')(sub2)
# drop2 = Dropout(0.1)(d2)
# d3 = Dense(units=1, activation='relu')(sub3)
# drop3 = Dropout(0.1)(d3)
#
# sub1 = Dense(units=16, activation='relu')(drop1)
# sub2 = Dense(units=16, activation='relu')(drop2)
# sub3 = Dense(units=16, activation='relu')(drop3)

out1 = Dense(units=1, activation='linear')(sub1)
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


def get_neighbor_weight(cell_id, nbh_distance=1):

    nb_df = pd.read_csv('data/nb_size_' + str(nbh_distance) + '_weight.csv')
    nb_df.dropna(inplace=True)

    nb_df = nb_df[nb_df['source'] == cell_id]
    # df = pd.read_csv('data/finalized_weight.csv')
    nb_weights = nb_df

    # we predict year 2001, 2002, 2003, hence these fields are not necessary
    nb_weights.drop('source', axis=1, inplace=True)
    nb_weights.drop('s_y00', axis=1, inplace=True)
    nb_weights.drop('s_y01', axis=1, inplace=True)
    nb_weights.drop('s_y02', axis=1, inplace=True)
    nb_weights.drop('s_y03', axis=1, inplace=True)
    nb_weights.drop('nb_size', axis=1, inplace=True)
    nb_weights.drop('neighbor', axis=1, inplace=True)
    nb_weights.drop('distance', axis=1, inplace=True)
    nb_weights.drop('nb_y00', axis=1, inplace=True)

    return nb_weights

# cell_index = 0
# def handle_sigle_instance(g_pred):
#     global cell_index
#
#     print('working...')
#     cell_index = cell_index + 1
#
#     return 0

def my_loss_function(y_true, y_pred):

    global grid_cells_id_test

    # compute mean square prediction and observation
    mean_square_loss = K.mean(K.square(y_pred - y_true), axis=-1)

    # weighted mean square with spatial concern
    lamda = 0.01
    neighbors_loss = np.array([], dtype='float32')
    row_index = 0
    nbh_distance = 1
    for group in grid_cells_id_test:
        neighbors = get_neighbor_weight(group, nbh_distance=nbh_distance)
        if neighbors.isnull().values.any():
            print('has nan', row_index)
            continue
        yield_indices = neighbors.columns[1:4]
        nb_yields = neighbors[yield_indices]
        source_yield = y_pred[row_index, :]
        nb_np_yields = nb_yields.values

        nb_yield_tensor = tf.convert_to_tensor(nb_np_yields, tf.float32)
        ## mean square error of three years
        square_error = tf.reduce_mean(K.square(source_yield - nb_yield_tensor), axis=1)

        nb_weights = neighbors['nb_weight']
        nb_weight_tensor = tf.convert_to_tensor(nb_weights.values, tf.float32)

        loss = lamda*K.sum(nb_weight_tensor*square_error)
        # loss = lamda*K.sum(nb_weight_tensor*tf.reduce_mean(nb_yield_tensor, axis=1))
        np.append(neighbors_loss, loss)
        row_index = row_index + 1


    # convert to tensor
    loss_tensor = tf.convert_to_tensor(neighbors_loss, tf.float32)
    return mean_square_loss + K.mean(loss_tensor, axis=-1)
    # return K.mean(y_pred - tf.reshape(loss_tensor, [-1, 3]), axis=-1)
    # return mean_square_loss

    # return mean_squared_error(y_true, y_pred)

# Compiling the model using 'adam' optimizer and MSE as loss function
# sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

# model.compile(optimizer=sgd, loss=my_loss_function,  metrics=['mse', 'mae', 'mape'],  loss_weights=[1.0, 1.0, 1.0])
model.compile(optimizer='adam', loss=my_loss_function,  metrics=['mse', 'mae', 'mape'],  loss_weights=[1.0, 1.0, 1.0])
# model.compile(optimizer='adam', loss='mean_squared_error',  metrics=['mse', 'mae', 'mape'],  loss_weights=[1.0, 1.0, 1.0])

#muti_outputs shape= tasks x train_samples
callbacks = []
model.fit(x=dat_train, y=[label_train_1, label_train_2, label_train_3], epochs=5000, batch_size=32)

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

mse = mean_squared_error(label_test_1, pred1.flatten())
print("test mse=", mse)
print("test mape=", mean_absolute_percentage_error(label_test_1, pred1.flatten()))
print("test mae=", mean_absolute_error(label_test_1, pred1.flatten()))
print("me=", max(abs(plot_y)))
plt.show()


## do prediction on entire dataset
pred2003, pred2002, pred2001 = model.predict(dat)
df['PRED_03'] = pred2003.flatten()
df['PRED_02'] = pred2002.flatten()
df['PRED_01'] = pred2001.flatten()

df.to_csv('output/prediction_f29_' + str(mse) + '.csv', index=False)