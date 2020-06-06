#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import MinMaxScaler
from matplotlib.gridspec import GridSpec
from progressbar import progressbar
from joblib import load
import os
import six
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# In[2]:


'''
Class ini berfungsi untuk menggambarkan dan menyimpan confusion matrix
pada suatu data. Confusion matrix yang ditampilkan bisa lebih dari satu
'''


class ConfusionMatrix:

    def __init__(self, figsize, num_rows, num_cols, labels):
        self.figsize = figsize
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.plots = []
        self.fig = None
        self.gs = None
        self.labels = labels
        self.xylabel_size = 12
        self.label_size = 12
        self.count_size = 12
        self.suptitle = None

    '''
    Menambahkan confusion matrix
    '''

    def add(self, Y_true, Y_pred, title, row_title='Kelas Sebenarnya', col_title='Kelas Prediksi', cmap=plt.cm.Blues):
        self.plots.append(
            [Y_true, Y_pred, title, row_title, col_title, cmap]
        )

    '''
    Menampilkan seluruh confusion matrix
    '''

    def show(self):
        self.fig = plt.figure(figsize=self.figsize, dpi=300)
        self.gs = GridSpec(self.num_rows, self.num_cols, figure=self.fig)
        self.fig.suptitle(self.suptitle, fontsize=20, va='center', ha='center')

        for index, subplot in enumerate(self.plots):
            self.draw_conf_mat(subplot, index + 1)

    '''
    Menggambar suatu confusion matrix
    '''

    def draw_conf_mat(self, data, index):
        Y_true = data[0]
        Y_pred = data[1]
        title = data[2]
        ylabel = data[3]
        xlabel = data[4]
        cmap = data[5]

        # Mendapatkan lokasi confusion matrix yang akan ditampilkan
        row_pos = int(np.ceil(index / self.num_cols))
        col_pos = int(index - (self.num_cols * (row_pos - 1)))

        conf_mat = confusion_matrix(Y_true, Y_pred)
        percentage = (conf_mat / conf_mat.sum(axis=1)[:, np.newaxis]) * 100

        axes = plt.subplot(self.gs[row_pos - 1, col_pos - 1])
        im = axes.imshow(conf_mat, interpolation='nearest', cmap=cmap)

        axes.set(
            xticks=np.arange(conf_mat.shape[1]),
            yticks=np.arange(conf_mat.shape[0]),
            xticklabels=self.labels,
            yticklabels=self.labels,
            ylabel=ylabel,
            xlabel=xlabel
        )

        axes.title.set_position([0.5, 1.1])
        axes.title.set_text(title)
        axes.title.set_size(15)
        axes.yaxis.label.set_size(self.xylabel_size)
        axes.xaxis.label.set_size(self.xylabel_size)

        # Rotate the tick labels and set their alignment.
        plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor", fontsize=self.label_size)
        plt.setp(axes.get_yticklabels(), fontsize=self.label_size)

        # Loop over data dimensions and create text annotations.
        thresh = conf_mat.max() / 2.
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                content = f'{conf_mat[i, j]} citra\n{percentage[i, j]:.2f}%'

                axes.text(j, i, content,
                          ha="center", va="center",
                          color="white" if conf_mat[i, j] > thresh else "black",
                          fontsize=self.count_size)

        akurasi = np.around(accuracy_score(Y_true, Y_pred) * 100, 2)
        axes.text(0.5, 1.05, f'Akurasi : {akurasi}%', fontsize=13, ha='center', transform=axes.transAxes)

    '''
    Menyimpan confusion matrix pada path tertentu
    '''

    def save(self, path, dpi=400):
        plt.savefig(path, bbox_inches='tight', dpi=dpi)


# In[3]:


x, y = load('features.jlb')
scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
del x_train
del y_train
del x
del y


# In[4]:


def cm(y_true, y_pred, title=''):
    cfm = ConfusionMatrix((10, 10), 1, 1, ['A', 'B', 'C'])
    cfm.count_size = 15
    cfm.xylabel_size = 15
    cfm.add(y_true, y_pred, title)
    cfm.save(title)
    return cfm.show()


# In[5]:


data = {}
for path in progressbar(os.listdir('model')):
    clf = load(f'model/{path}')
    y_pred = clf.predict(x_test)
    data['neighbor = ' + path.split('.')[0].split('=')[1]] = accuracy_score(y_test, y_pred)
    cm(y_test, y_pred, path.split('.')[0])


# In[7]:


def plot(data, maxt=10, title=None, fz=(10, 5)):
    key = np.array(list(data.keys()))
    val = np.array(list(data.values()))
    sortidx = (np.argsort(val)[::-1])[:maxt]
    key = key[sortidx]
    val = val[sortidx]
    val = val * 100
    val = np.around(val, 2)

    y_pos = np.arange(key.shape[0])

    fig, ax = plt.subplots(figsize=fz)
    plt.bar(y_pos, val, align='center', alpha=0.5)
    plt.xticks(y_pos, key, rotation=60)
    plt.yticks(np.arange(0, 101, 20))
    plt.ylabel('Akurasi dalam %')
    plt.xlabel('Model KNN')
    for index, data in enumerate(val):
        plt.text(x=index - 0.35, y=data + 0.75, s=f'{data}%', fontdict=dict(fontsize=fz[0]))
    if title is not None:
        plt.title(title)
    return plt.show()


plot(data, title='Perbandingan model KNN')