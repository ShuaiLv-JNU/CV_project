"""
desc: 可视化训练过程
"""
import numpy as np


def load_file(filename):
    """

    :param filename:
    :return:
    """
    import pickle
    data_file = open(filename, 'rb')
    data = pickle.load(data_file)
    data_file.close()
    return data.history


def plot_loss(his, ds):
    """

    :param his:
    :return:
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(his.history['loss'])), his.history['loss'], label='train loss')
    plt.plot(np.arange(len(his.history['val_loss'])), his.history['val_loss'], label='valid loss')
    plt.title(ds + ' training loss')
    plt.legend(loc='best')
    plt.savefig(r'D:\PycharmProjects\DL\assets\his_loss.png')
    plt.show()

def plot_acc(his, ds):
    """

    :param his:
    :return:
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(his.history['accuracy'])), his.history['accuracy'], label='train accuracy')
    plt.plot(np.arange(len(his.history['val_accuracy'])), his.history['val_accuracy'], label='valid accuracy')
    plt.title(ds + ' training accuracy')
    plt.legend(loc='best')
    plt.savefig(r'D:\PycharmProjects\DL\assets\his_acc.png')
    plt.show()

def get_feature_map(model, layer_index, channels, input_img):
    from tensorflow.python.keras import backend as K
    layer = K.function([model.layers[0].input], [model.layers[layer_index].output])
    feature_map = layer([input_img])[0]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 8))
    for i in range(channels):
        img = feature_map[:, :, :, i]
        plt.subplot(4, 8, i + 1)
        plt.imshow(img[0], cmap='gray')
    plt.savefig(r'D:\PycharmProjects\DL\dataset\0-f.jpg')
    plt.show()

def plot_feature_map():
    from model import CNN3
    model = CNN3()
    model.load_weights(r'D:\PycharmProjects\DL\models\cnn3_best_weights.h5')
    import cv2
    img = cv2.cvtColor(cv2.imread(r'D:\PycharmProjects\DL\dataset\0.jpg'), cv2.COLOR_BGR2GRAY)
    img.shape = (1, 48, 48, 1)
    get_feature_map(model, 4, 32, img)
if __name__ == '__main__':
    from model import CNN3
    model = CNN3()
    model.load_weights(r'D:\PycharmProjects\DL\cnn3_best_weights.h5')
    import cv2
    img = cv2.cvtColor(cv2.imread(r'D:\PycharmProjects\DL\dataset\0.jpg'), cv2.COLOR_BGR2GRAY)
    img.shape = (1, 48, 48, 1)
    get_feature_map(model, 6, 64,img)