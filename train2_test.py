import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_utils import shuffle, to_categorical
import numpy as np
k = 12

# Depth (40, 100, ...)
L = 40
nb_layers = int((L - 4) / 3)
# Data loading and preprocessing
X=np.load('/home/zhb/NewDisk/chenruoxi/nsfw-defense/DataSet/images10000.npy')
Y=np.load('/home/zhb/NewDisk/chenruoxi/nsfw-defense/DataSet/labels10000.npy')
# X, Y = shuffle(X, Y)
Y = np.reshape(Y, newshape=[10000])
Y = to_categorical(Y, nb_classes=5)

print(X.shape)
print(Y.shape)
def inference(x):
    n=5
    net = tflearn.conv_2d(x, 16, 3, regularizer='L2', weight_decay=0.0001)
    net = tflearn.resnext_block(net, n, 16, 32)
    net = tflearn.resnext_block(net, 1, 32, 32, downsample=True)
    net = tflearn.resnext_block(net, n-1, 32, 32)
    net = tflearn.resnext_block(net, 1, 64, 32, downsample=True)
    net = tflearn.resnext_block(net, n-1, 64, 32)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    # Regression
    net = tflearn.fully_connected(net, 5, activation=None)
    logits = net
    net = tflearn.activations.softmax(net)
    return logits, net

x = tflearn.input_data(shape=[None, 128, 128, 3])
logits, net  = inference(x)
model = tflearn.DNN(net)

# model.fit(X, Y, validation_set=0.2, n_epoch=150, shuffle=True,
#           show_metric=True, batch_size=8)

model.load('/home/zhb/NewDisk/chenruoxi/nsfw-defense/train-model2/nsfw_model_8230')

# index1 = [1, 1001, 2001, 3001, 4001]
# test_X = X[index1]
# test_Y = Y[index1]
test_X = np.load('/home/zhb/NewDisk/chenruoxi/nsfw-defense/DataSet/deepfool-test')
confP = model.predict(test_X.reshape([-1, 128, 128, 3]))
# print('test_Y:', np.argmax(test_Y, 1))
print('pre label:', np.argmax(confP, 1))
print('pre conf:', (confP*10000).astype(np.int64)/100)











