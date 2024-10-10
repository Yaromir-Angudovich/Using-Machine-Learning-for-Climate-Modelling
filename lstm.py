# Загрузка необходимых библиотек
import scipy.io  # Библиотека для загрузки файла
import numpy as np  # Для работы с массивами
from keras.models import Sequential  # Для создания модели
import matplotlib.font_manager as font_manager
import pandas as pd  # Для работы с датафреймами
from matplotlib import pyplot as plt
from matplotlib import pylab  # Для отрисовки графиков
from sklearn.preprocessing import MinMaxScaler  # Для масштабирования данных
from keras.layers import Dense, Dropout, LSTM, Bidirectional, BatchNormalization, \
    SimpleRNN  # Загрузка слоев для нейронной модели
from scipy.signal import savgol_filter
from matplotlib.ticker import FormatStrFormatter

mat = scipy.io.loadmat('testdata.mat')

X = mat['X']
Y = mat['Y']
Z = mat['Z']

t = mat['t']
x = mat['x']
y = mat['y']
z = mat['z']

X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
Z = Z.reshape(-1, 1)
t = t.reshape(-1, 1)
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
z = z.reshape(-1, 1)

X = X[500:-1000]
Y = Y[500:-1000]
Z = Z[500:-1000]
x = x[500:-1000]
y = y[500:-1000]
z = z[500:-1000]
t = t[500:-1000]

t[:] = (t[:] - 500) / 100

fig = plt.subplot(1, 1, 1)
plt.plot(t[:], z[:], 'b', label='z',
         linewidth=0.75)
plt.plot(t[:], Z[:], 'r', label='Z',
         linewidth=1.5)
font = font_manager.FontProperties(style='italic', size=22)
box = fig.get_position()
fig.set_position([box.x0, box.y0*1.2, box.width * 0.7, box.height * 1.05])
plt.yticks(np.arange(5, 40, 10))
fig.set_xlim([min(t[:]), max(t[:])])
plt.ylabel('Amplitude', fontsize=22)
plt.xlabel('Time, MTU', fontsize=22)
plt.legend(loc='upper center', frameon=False, prop=font, ncol=2)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
for axis in ['top', 'bottom', 'left', 'right']:
    fig.spines[axis].set_linewidth(0.75)

plt.show()

n_future = 50
n_past = 50
delta = -(int(X.shape[0] * 0.2))

X_data = np.hstack([X, Y, Z, x, y, z])
Y_data = np.hstack([x, y, z])
print(X_data.shape)
print(Y_data.shape)

scaler1 = MinMaxScaler(feature_range=(0, 1))
X_sc = scaler1.fit_transform(X_data)

scaler2 = MinMaxScaler(feature_range=(0, 1))
Y_sc = scaler2.fit_transform(Y_data)

X_scaled = X_sc[:delta]
x_val_scaled = Y_sc[:delta]
real_scaled_x = X_sc[delta:]
real_scaled_y = Y_sc[delta:]

print(delta)
MSE = np.zeros((n_future))

trainX = []
trainY = []

for i in range(n_past, len(X_scaled)):
    trainX.append(X_scaled[i - n_past:i, 0:X_scaled.shape[1]])
    trainY.append(Y_sc[i:i + n_future])

trainX = np.array(trainX)
trainY = np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

model = Sequential()
model.add(LSTM(units=6, return_sequences=True, input_shape=(n_past, 6)))
model.add(LSTM(units=128, return_sequences=True))
model.add(LSTM(units=128, return_sequences=True))
model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
# model.add(LSTM(units=128, return_sequences=True))
model.add(Dense(3))


model.compile(optimizer='adam', loss='MAE')

model.load_weights("weightsLSTM_50_128_3_BiLSTM.weights.h5")
# model.load_weights("weightsLSTM_50_128_3_LSTM.weights.h5")

df = pd.read_excel('history_50_128_3_BiLSTM.xlsx')
# df = pd.read_excel('history_50_128_3_LSTM.xlsx')
loss = df['Learning']
val_loss = df['Validation']
loss = loss.values.reshape(-1, 1)
val_loss = val_loss.values.reshape(-1, 1)
loss = savgol_filter(loss[:, 0], 9, 4)
val_loss = savgol_filter(val_loss[:, 0], 9, 4)

fig = plt.subplot(1, 1, 1)
plt.plot(loss, 'b', label='Learning', linewidth=1.5)
plt.plot(val_loss, 'r', label='Validation', linewidth=1.5)
plt.ylabel('Loss function', fontsize=22)
plt.xlabel('Epoch', fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
font = font_manager.FontProperties(style='italic', size=22)
box = fig.get_position()
fig.set_position([box.x0, box.y0*1.2, box.width * 0.7, box.height * 1.05])
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), frameon=False, prop=font, ncol=2)
for axis in ['top', 'bottom', 'left', 'right']:
    fig.spines[axis].set_linewidth(1.5)

plt.show()

n = len(X_scaled)
X_test = []
for i in range(n_past, int(n)):
    X_test.append(X_scaled[i - n_past:i, 0:X_scaled.shape[1]])
X_test = np.array(X_test)

predicted = model.predict(X_test[:])

n = len(real_scaled_x)
real_valid_x = []
real_valid_y = []
for i in range(n_past, int(n) - n_future):
    real_valid_x.append(real_scaled_x[i - n_past:i, 0:real_scaled_x.shape[1]])
    real_valid_y.append(real_scaled_y[i:i + n_future])
real_valid_x = np.array(real_valid_x)
real_valid_y = np.array(real_valid_y)

Y_Validation = model.evaluate(real_valid_x, real_valid_y)

Y_Forecast = model.predict(real_valid_x[:, :, :])

for i in range(Y_Forecast.shape[0]):
    Y_Forecast[i, :, :] = scaler2.inverse_transform(Y_Forecast[i, :, :])

real = scaler1.inverse_transform(real_scaled_x)

for i in range(predicted.shape[0]):
    predicted[i, :, :] = scaler2.inverse_transform(predicted[i, :, :])

fig = plt.subplot(3, 1, 1)
plt.plot(t[delta:], real[:, 3], 'r', label='True results',
         linewidth=1.5)
plt.plot(t[delta + 2 * n_past:], Y_Forecast[:, n_future - 1, 0], '--b', label='Predictions', linewidth=1.5)
font = font_manager.FontProperties(style='italic', size=22)
box = fig.get_position()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), frameon=False, prop=font, ncol=2)
plt.xticks(np.arange(min(t[delta + 2 * n_past:]), max(t[delta + 2 * n_past:]), 2.5))
plt.xticks(np.arange(68, 85, 2))
fig.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig.set_xlim([min(t[delta:]), max(t[delta:])])
plt.ylabel('x', fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
for axis in ['top', 'bottom', 'left', 'right']:
    fig.spines[axis].set_linewidth(1.5)

fig2 = plt.subplot(3, 1, 2)
plt.plot(t[delta:], real[:, 4], 'r',
         linewidth=1.5)
plt.plot(t[delta + 2 * n_past:], Y_Forecast[:, n_future - 1, 1], '--b', linewidth=1.5)
font = font_manager.FontProperties(style='italic', size=22)
box = fig2.get_position()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), frameon=False, prop=font, ncol=2)
plt.xticks(np.arange(min(t[delta + 2 * n_past:]), max(t[delta + 2 * n_past:]), 2.5))
plt.xticks(np.arange(68, 85, 2))
fig2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig2.set_xlim([min(t[delta:]), max(t[delta:])])
plt.ylabel('y', fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
for axis in ['top', 'bottom', 'left', 'right']:
    fig2.spines[axis].set_linewidth(1.5)
plt.subplots_adjust(wspace=0.5, hspace=0.5)

fig3 = plt.subplot(3, 1, 3)
plt.plot(t[delta:], real[:, 5], 'r',
         linewidth=1.5)
plt.plot(t[delta + 2 * n_past:], Y_Forecast[:, n_future - 1, 2], '--b', linewidth=1.5)
font = font_manager.FontProperties(style='normal', size=22)
box = fig3.get_position()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4), frameon=False, prop=font, ncol=2)
plt.xticks(np.arange(min(t[delta + 2 * n_past:]), max(t[delta + 2 * n_past:]), 2.5))
plt.xticks(np.arange(68, 85, 2))
fig3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
fig3.set_xlim([min(t[delta:]), max(t[delta:])])
plt.ylabel('z', fontsize=22)
plt.xlabel('Time, MTU', fontsize=22)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
for axis in ['top', 'bottom', 'left', 'right']:
    fig3.spines[axis].set_linewidth(1.5)
plt.subplots_adjust(wspace=0.5, hspace=0.5)
fig.set_position([box.x0, box.y0*6.2, box.width * 0.7, box.height * 1.05])
fig2.set_position([box.x0, box.y0*3.7, box.width * 0.7, box.height * 1.05])
fig3.set_position([box.x0, box.y0*1.2, box.width * 0.7, box.height * 1.05])
fig.set_yticks(np.arange(-15, 15.1, 15))
fig2.set_yticks(np.arange(-15, 15.1, 15))
fig3.set_yticks(np.arange(0, 40.1, 20))

plt.show()
