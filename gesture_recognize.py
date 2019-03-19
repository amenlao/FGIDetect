# -*- coding: utf-8 -*-
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
import time

# 可以通过下面的代码主动创建一个使用了ConfigProto的Session，再注入到Keras的后端中去
config = tf.ConfigProto()
# 避免默认将显存吃满
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

import numpy as np
import matplotlib.pyplot as plt
import os
# import theano
from PIL import Image
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json

import cv2
import matplotlib
# matplotlib.use("TkAgg")
#from matplotlib import pyplot as plt
# 通过引入这个backend，就可以让Keras来处理兼容性，将后端的名字设为K
from keras import backend as K

K.set_image_dim_ordering('th')  # theano

WeightFileName = ["binmask_CnnModel.hdf5", "skinmask_CnnModel.hdf5"]
#"skin_mask_CnnModel.hdf5"]
label_dict = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
              'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ok': 10}

# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = 1

# Number of epochs to train (change it accordingly)
nb_epoch = 15  # 25

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

## NOTE: If you change this then dont forget to change Labels accordingly
# nb_classes = 5
nb_classes = 11

# Batch_size to train
batch_size = 32


class GestureRecognize():
    def getModel(self):
        self.model = self.loadCNN(0)
        return self.model

    def __init__(self, wf_index=0, parent=None):
        if wf_index == 0:
            self.model = self.getModel()
            print('-----GestureRecognize----')
            print(self.model)
        else:
            self.wf_index = wf_index
            self.model = None
            print('-----GestureRecognize( parameter ..)----')

    def export_cnn_net_framework(self):
        if self.model == None:
            print('model get failed!')
            return
        # Model summary
        self.model.summary()
        # Model conig details
        self.model.get_config()

        from keras.utils import plot_model

        model_net_framework_name = 'cnn_model_' + time.strftime('%Y-%m-%d_%H-%M-%S',
                                                                time.localtime(time.time())) + '.png'
        plot_model(self.model, to_file=model_net_framework_name, show_shapes=True)
        print('except sucessfully!')

    def loadCNN(self, wf_index, weight_name='binmask_CnnModel.hdf5'):
        model = Sequential()
        model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                         padding='valid',
                         # input_shape=( img_rows, img_cols,img_channels)))
                         input_shape=(img_channels, img_rows, img_cols)))  # theano
        convout1 = Activation('relu')
        model.add(convout1)
        model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
        convout2 = Activation('relu')
        model.add(convout2)
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        if wf_index >= 0:
            fname = WeightFileName[wf_index]
            print("loading ", fname)
            # 加载预训练的权重，来重新实例化你的模型
            model.load_weights(fname)
        else:
            print('使用自定义网络 !')
            model.load_weights(weight_name)
            # binmask_CnnModel.hdf5
        layer = model.layers[11]
        self.model = model
        return model

    def static_video_recognize(self, img):
        if self.model == None:
            print('model get failed!')
            return
        image = np.array(img).flatten()
        # reshape it
        image = image.reshape(img_channels, img_rows, img_cols)
        # float32
        image = image.astype('float32')
        # reshape for NN
        rimage = image.reshape(1, img_channels, img_rows, img_cols)  # theano
        prob_array = self.model.predict_proba(rimage)
        # print (prob_array)

        d = {}
        i = 0
        output = list(label_dict.keys())
        for items in output:
            d[items] = prob_array[0][i] * 100
            i += 1

        # Get the output with maximum probability
        import operator
        # 求出概率最大值
        guess = max(d.items(), key=operator.itemgetter(1))[0]
        prob = d[guess]
        return guess, prob

    def static_image_recognize(self, imgPng):
        # print ('===============static_image_recognize==============')
        if self.model == None:
            print('model get failed!')
            return
        image = np.array(Image.open(imgPng).convert('L')).flatten()
        # Load image and flatten it
        # image = np.array(img).flatten()

        # reshape it
        image = image.reshape(img_channels, img_rows, img_cols)

        # float32
        image = image.astype('float32')

        # normalize it
        image = image / 255

        # reshape for NN
        rimage = image.reshape(1, img_channels, img_rows, img_cols)  # theano
        # print (rimage.shape)
        prob_array = self.model.predict_proba(rimage)

        d = {}
        i = 0
        output = list(label_dict.keys())
        for items in output:
            d[items] = prob_array[0][i] * 100
            i += 1

        # Get the output with maximum probability
        import operator
        # 求出概率最大值
        guess = max(d.items(), key=operator.itemgetter(1))[0]
        prob = d[guess]
        return guess

    def getImgListPath(self, train_set_path, imlist):
        all_path = os.listdir(train_set_path)
        for f in all_path:
            p = os.path.join(train_set_path, f)
            if os.path.isdir(p):
                self.getImgListPath(p, imlist)
            elif os.path.isfile(p):
                if os.path.splitext(p)[1] == '.png':
                    imlist.append(p)

    def visualizeLayers(self, imgName, layerIndex):
        model = self.model
        if model is None:
            model = self.loadCNN(0)

        image = np.array(Image.open(imgName).convert('L')).flatten()

        ## Predict
        # guessGesture(model,image)

        # reshape it
        image = image.reshape(img_channels, img_rows, img_cols)

        # float32
        image = image.astype('float32')

        # normalize it
        image = image / 255

        # reshape for NN
        input_image = image.reshape(1, img_channels, img_rows, img_cols)

        # visualizing intermediate layers
        # output_layer = model.layers[layerIndex].output
        # output_fn = theano.function([model.layers[0].input], output_layer)
        # output_image = output_fn(input_image)

        if layerIndex >= 1:
            self.visualizeLayer(model, imgName, input_image, layerIndex)
        else:
            tlayers = len(model.layers[:])
            print("Total layers - {}".format(tlayers))
            for i in range(1, tlayers):
                self.visualizeLayer(model, imgName, input_image, i)

    # %%
    def visualizeLayer(self, model, imgName, input_image, layerIndex):
        layer = model.layers[layerIndex]

        get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output, ])
        activations = get_activations([input_image, 0])[0]
        output_image = activations

        ## If 4 dimensional then take the last dimension value as it would be no of filters
        if output_image.ndim == 4:
            # Rearrange dimension so we can plot the result
            o1 = np.rollaxis(output_image, 3, 1)
            output_image = np.rollaxis(o1, 3, 1)

            print("Dumping filter data of layer{} - {}".format(layerIndex, layer.__class__.__name__))
            filters = len(output_image[0, 0, 0, :])

            fig = plt.figure(figsize=(8, 8))
            # This loop will plot the 32 filter data for the input image
            for i in range(filters):
                ax = fig.add_subplot(6, 6, i + 1)
                # ax.imshow(output_image[img,:,:,i],interpolation='none' ) #to see the first filter
                ax.imshow(output_image[0, :, :, i], 'gray')
                # ax.set_title("Feature map of layer#{} \ncalled '{}' \nof type {} ".format(layerIndex,
                #                layer.name,layer.__class__.__name__))
                plt.xticks(np.array([]))
                plt.yticks(np.array([]))
            plt.tight_layout()
            plt.show()
            imgName = os.path.basename(imgName).split('.')[0]
            fig.savefig("img_" + imgName + "_layer" + str(layerIndex) + "_" + layer.__class__.__name__ + ".png")
            plt.close(fig)
        else:
            print("Can't dump data of this layer{}- {}".format(layerIndex, layer.__class__.__name__))


if __name__ == '__main__':
    g = GestureRecognize()
    imlist = []
    g.getImgListPath('train_set', imlist)
    g.visualizeLayers(imgName='train_set/eight/eight_25.png', layerIndex=3)
    os.system('pause')
    for imgName in imlist:
        # image = np.array(Image.open('./imgs/' + imgName).convert('L')).flatten()
        guess = g.static_image_recognize(imgName)
        print('src:{}---recognizaiton:{}'.format(imgName, guess))
