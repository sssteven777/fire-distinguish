import numpy as np
from tensorflow import keras
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

def to_training_data(train_data, data):
    for i in range(len(data)):
        for j in range(6):
            if j <= 5 and i < cases:
                train_data[i][j] = data[i][j]
    return train_data

def to_training_label(train_label, data):
    for i in range(len(data)):
        train_label[i] = data[i][6]
    return train_label


def to_one_hot(train_label):
    results = np.zeros((cases, cate))
    for i, x in enumerate(train_label):
        results[i, x-1] = 1
    return results


def Normalization(train_data):
    for i in range(len(train_data)):
        train_data[i][0] = (train_data[i][0] - 50) / (700)
        train_data[i][1] = (train_data[i][1] - 25) / (325)
        train_data[i][2] = (train_data[i][2] - 1) / (49)
        train_data[i][3] = (train_data[i][3] - 10) / (9)
        train_data[i][4] = (train_data[i][4] - 2) / (7)
        train_data[i][5] = (train_data[i][5] - 100) / (1920)

    return train_data


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(10, activation='sigmoid', input_shape=(6,)))
    model.add(layers.Dense(3, activation='sigmoid'))

    model.compile(
        optimizer=optimizers.RMSprop(lr=0.03),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


cases = 60
features = 6
cate = 3
k = 4

p = 'fire.csv'
with open(p, encoding='utf-8') as f:
    data = np.loadtxt(f, str, delimiter=",")
data1 = data[1:]
np.random.shuffle((data1))

train_data_1 = np.zeros(shape=(cases, features))
train_label_1 = np.zeros(shape=(cases))

train_data = to_training_data(train_data_1, data1)
train_label = to_training_label(train_label_1, data1)
train_label = train_label.astype('int32')
train_label_vec = to_one_hot(train_label)

train_data = Normalization(train_data)
num_val_samples = len(train_data) // k
num_epochs = 30
all_trainacc_histiries = []
all_valacc_histories = []
for i in range(k):
    print('processing fold #', i+1)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_label = train_label_vec[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i+1) * num_val_samples:]],
        axis=0
    )
    partial_train_label = np.concatenate(
        [train_label_vec[:i * num_val_samples],
         train_label_vec[(i+1) * num_val_samples:]],
        axis=0
    )
    model = build_model()

    history = model.fit(
        partial_train_data,
        partial_train_label,
        epochs=num_epochs,
        batch_size=1,
        validation_data=(val_data, val_label)
    )
    acc_history = history.history['accuracy']
    val_acc_history = history.history['val_accuracy']
    all_trainacc_histiries.append(acc_history)
    all_valacc_histories.append(val_acc_history)


average_trainacc_history = [
   np.mean([x[i] for x in all_trainacc_histiries]) for i in range(num_epochs)
]
average_valacc_history = [
    np.mean([x[i] for x in all_valacc_histories]) for i in range(num_epochs)
]

plt.plot(range(1, num_epochs+1), average_trainacc_history, 'bo', label='Training_acc')
plt.plot(range(1, num_epochs+1), average_valacc_history, 'b', label='val_acc')
plt.title('Training acc and val acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.show()
