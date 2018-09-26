from keras.engine.input_layer import Input
from keras.engine.training import Model
from keras.layers.core import Dense

import numpy as np
import matplotlib.pyplot as plt

n_row = 1000
x1 = np.random.randn(n_row)
x2 = np.random.randn(n_row)
x3 = np.random.randn(n_row)
dat = np.array([x1, x2, x3]).transpose()

# Generate indexes of test and train
idx_list = np.linspace(0, 999, num=1000)
idx_test = np.random.choice(n_row, size=200, replace=False)
idx_train = np.delete(idx_list, idx_test).astype('int')

# Split data into test and train
dat_train = dat[idx_train, :]
dat_test = dat[idx_test, :]


x = Input(shape=(3, ))
shared = Dense(units=32)(x)
sub1 = Dense(units=16, activation='relu')(shared)
sub2 = Dense(units=16, activation='relu')(shared)
sub3 = Dense(units=16, activation='relu')(shared)

out1 = Dense(units=1, activation='linear')(sub1)
out2 = Dense(units=1, activation='linear')(sub2)
out3 = Dense(units=1, activation='linear')(sub3)


model = Model(inputs=x, outputs=[out1, out2, out3])


label1 = np.random.rand(800, 1)
label2 = np.random.rand(800, 1)
label3 = np.random.rand(800, 1)

label_test = np.random.rand(200, 1)

# Compiling the model using 'adam' optimizer and MSE as loss function
model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x=dat_train, y=[label1, label2, label3], epochs=5, batch_size=20)

pred1, pred2, pred3 = model.predict(dat_test)

plot_x = dat_test[:, 0]
plot_y = np.array(pred1) - label_test
plt.scatter(x=dat_test[:, 0], y=np.array(pred1) - label_test)

plt.show()
