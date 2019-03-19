# -*- coding: utf-8 -*-
# 第一行必须直接import这个函数，而非keras这个包，而且这句话必须在其他import keras之前，
# 否则keras初始化之后，再替换一个session，原session也不会释放
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import numpy as np
from keras import backend as K

WeightFileName = ["myNewCnnModel.hdf5"]
# class names
label_dict = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
              'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ok': 10}

# 可以通过下面的代码主动创建一个使用了ConfigProto的Session，再注入到Keras的后端中去
config = tf.ConfigProto()
# 避免默认将显存吃满
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import np_utils
from keras.callbacks import Callback

# import matplotlib.pyplot as plt
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
import matplotlib.pyplot as plt

# 通过引入这个backend，就可以让Keras来处理兼容性，将后端的名字设为K
from keras import backend as K


K.set_image_dim_ordering('th')  # theano

# input image dimensions
img_rows, img_cols = 200, 200

# number of channels
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = 1

# Number of epochs to train (change it accordingly)
nb_epoch = 35  #5  #15  # 25

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2
# Size of convolution kernel
nb_conv = 3

## NOTE: If you change this then dont forget to change Labels accordingly
nb_classes = 11

## train_set_path is the folder which is fed in to training model
train_set_path = 'train_set2'

# Batch_size to train
batch_size = 32

#写一个LossHistory类，保存loss和acc
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()


class TrainGestureModeByCNN:
    def __init__(self, parent=None):
        # self.model = self.loadCNN( wf_index = 0)
        self.model = None
        # 创建一个实例history
        self.history = LossHistory()

    def write_TFRecords(self, tfr_file_name='train.tfrecords', img_rows=200, img_cols=200, img_channels=1):
        imlist = []
        self.getImgListPath(train_set_path, imlist)
        num = len(imlist)
        writer = tf.python_io.TFRecordWriter(tfr_file_name)
        print('%d imgs!' % (num))

        for i, img_file in enumerate(imlist):
            img = Image.open(img_file)
            arr = np.asarray(img, dtype="float32")
            arr = arr.reshape(img_rows * img_cols, )

            img_raw = arr.tobytes()
            index = label_dict[os.path.basename(img_file).split('_')[0]]

            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
            writer.write(example.SerializeToString())
        writer.close()

    def read_TFRecords(self, tfr_file_name='train.tfrecords', img_rows=200, img_cols=200, img_channels=1):
        filename_queue = tf.train.string_input_producer([tfr_file_name])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'label': tf.FixedLenFeature([], tf.int64),
                'img_raw': tf.FixedLenFeature([], tf.string)
                # 'img_rows': tf.FixedLenFeature([], tf.int64),
                # 'img_cols': tf.FixedLenFeature([], tf.int64),
            }
        )
        # label
        label = features['label']
        image = features['img_raw']
        image = tf.decode_raw(image, tf.float32)

        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5  # normalize
        image = tf.reshape(image, [img_rows, img_rows, ])  # image
        label = tf.cast(label, tf.int32)  # tf.reshape(image,  tf.stack([img_rows, img_cols, 1]))

        print(image.shape)  # 可以做一些预处理之类的
        return image, label  # print(label)

    def read_SimpleTFRecords(self, tfr_file_name="train.tfrecords"):
        image_list = []
        label_list = []
        index = 0
        for serialized_example in tf.python_io.tf_record_iterator("train.tfrecords"):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)

            image = example.features.feature['img_raw'].bytes_list.value

            label = example.features.feature['label'].int64_list.value
            # 可以做一些预处理之类的
            # print(image)
            # print(label)
            image_list.insert(index, image)
            label_list.insert(index, label)
            index += 1
        print('%d imgs in tfr files!' % (index))
        return image_list, label_list

    def test_ReadRFRcord(self):
        # make_TFRecords()
        # os.system('pause')
        # img, label = self.read_TFRecords(tfr_file_name="train.tfrecords")
        # img, label = self.read_and_decode(tfr_file_name="train.tfrecords")
        img, label = self.read_SimpleTFRecords(tfr_file_name="train.tfrecords")

        os.system('pause')
        # 使用shuffle_batch可以随机打乱输入
        img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=30, capacity=60, min_after_dequeue=30)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for i in range(30):
                val, l = sess.run([img_batch, label_batch])
                # 我们也可以根据需要对val， l进行处理
                # l = to_categorical(l, 12)
                print(val.shape, l)
            coord.request_stop()
            coord.join(threads)

    def createCNNModel(self):
        self.model = Sequential()

        self.model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                              padding='valid',
                              # input_shape=( img_rows, img_cols,img_channels)))
                              input_shape=(img_channels, img_rows, img_cols)))  # theano
        convout1 = Activation('relu')
        self.model.add(convout1)
        self.model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
        convout2 = Activation('relu')
        self.model.add(convout2)
        self.model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))

        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        # Model summary
        self.model.summary()
        # Model conig details
        self.model.get_config()

        from keras.utils import plot_model
        plot_model(self.model, to_file='my_model.png', show_shapes=True)
        return self.model

    def getImgListPath(self, train_set_path, imlist):
        all_path = os.listdir(train_set_path)
        for f in all_path:
            p = os.path.join(train_set_path, f)
            if os.path.isdir(p):
                self.getImgListPath(p, imlist)
            elif os.path.isfile(p):
                if os.path.splitext(p)[1] == '.png':
                    imlist.append(p)

    def trainModel(self, train_set_path, weight_name):
        self.model = self.createCNNModel()
        imlist = []
        self.getImgListPath(train_set_path, imlist)

        image1 = np.array(Image.open(imlist[0]))
        m, n = image1.shape[0:2]
        total_images = len(imlist)

        img_ndarry_list = []

        label_list = np.ones((total_images,), dtype=int)

        for index, img_file in enumerate(imlist):
            single_img_label = label_dict[os.path.basename(img_file).split('_')[0]]

            single_img_array = np.array(Image.open(img_file).convert('L')).flatten()
            img_ndarry_list.insert(index, single_img_array)
            label_list[index] = single_img_label

        img_matrix = np.array(img_ndarry_list, dtype='f')
        data, label = shuffle(img_matrix, label_list, random_state=2)
        train_data = [data, label]

        (X, y) = (train_data[0], train_data[1])

        # Split X and y into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

        X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # normalize
        X_train /= 255
        X_test /= 255

        # convert integers to dummy variables (one hot encoding)
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        # print ( X_train, X_test, Y_train, Y_test)
        # print (img_matrix)

        hist = self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                              verbose=1, validation_split=0.2, validation_data=(X_test, Y_test),
                              callbacks =[self.history])

        # 保存模型的权重
        self.model.save(weight_name)

        # 模型评估
        score = self.model.evaluate(X_test, Y_test, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        # 绘制acc-loss曲线
        #self.history.loss_plot('epoch')
        print('train model success!')

    def static_image_recognize(self, img):
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
        print ('guess:'+guess)
        return guess

    def batch_test_static_image_recognize(self, imgs_set='train_set'):
        self.loadCNN(0)
        imlist = []
        self.getImgListPath(imgs_set, imlist)
        for imgName in imlist:
            # image = np.array(Image.open('./imgs/' + imgName).convert('L')).flatten()
            guess = self.static_image_recognize(imgName)
            print('src:{}---recognizaiton value:{}'.format(imgName, guess))

    def loadCNN(self, wf_index):
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

        # Model summary
        model.summary()
        # Model conig details
        model.get_config()

        from keras.utils import plot_model
        plot_model(model, to_file='new_model.png', show_shapes=True)
        if wf_index >= 0:
            fname = WeightFileName[wf_index]
            print("loading ", fname)
            # 加载预训练的权重，来重新实例化你的模型
            model.load_weights(fname)
        else:
            print('used to train model!')
            return None
        self.model = model
        return model



if __name__ == '__main__':
    train_net_obj = TrainGestureModeByCNN()
    # train_net_obj.trainModel(train_set_path = 'train_set',weight_name='myNewCnnModel2.hdf5')
    # train_net_obj.write_TFRecords(tfr_file_name='train.tfrecords',img_rows=200, img_cols=200,img_channels=1)
    # train_net_obj.test_ReadRFRcord()
    # train_net_obj.loadCNN( wf_index=0)
    # train_net_obj.static_image_recognize(imgPng='train_set/eight/eight_24.png')


    # train_net_obj.batch_test_static_image_recognize( imgs_set='train_set')
