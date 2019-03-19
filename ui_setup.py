# -*- coding: utf-8 -*-
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtSql import *
import os
import sys
import copy
from PIL import Image
import time
import cv2
import qimage2ndarray
import mmap
import numpy as np
import multiprocessing
import re
import random
import signal
import camera_face as camera

sample_counts = 400
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


# 工具类（单例模式）
class UtilOperation(object):
    _shared_state = {}
    mGesture = None
    mm = None

    def __new__(cls, *args, **kwargs):
        obj = super(UtilOperation, cls).__new__(cls, *args, **kwargs)
        obj.__dict__ = cls._shared_state
        print('__new__')
        return obj

    def gestureLoadNNByBtn(self, weight_index=0, weight_name='binmask_CnnModel.hdf5'):
        # xunfengd
        # from   train_gesture_model import TrainGestureModeByCNN
        from   gesture_recognize import GestureRecognize
        from keras import backend as K
        K.clear_session()  # clear session

        # xunfengd get gesture object
        self.mGesture = GestureRecognize(-1)
        self.mGesture.loadCNN(weight_index, weight_name)
        self.mGesture.static_image_recognize('train_set/eight/eight_24.png')
        print('gestureLoadNN...' + weight_name)
        return self.mGesture

    def gestureLoadNN(self, weight_index=0, weight_name='binmask_CnnModel.hdf5'):
        if self.mGesture is None:
            from   gesture_recognize import GestureRecognize
            from keras import backend as K
            K.clear_session()  # clear session

            self.mGesture = GestureRecognize()
            self.mGesture.loadCNN(weight_index, weight_name)
            # 手动预测一次，避免keras报错
            self.mGesture.static_image_recognize('train_set/eight/eight_24.png')
            print('gestureLoadNN...' + weight_name)
            return self.mGesture

    def getMmap(self):
        if self.mm == None:
            self.mm = mmap.mmap(fileno=-1, length=1024 * 1024, access=mmap.ACCESS_READ, tagname='share_mmap')
        return self.mm


# 训练CNN网络线程
class TrainCNNThread(QThread):
    sendlog = pyqtSignal(object)

    def __init__(self, mUtil, parent=None):
        super(TrainCNNThread, self).__init__(parent)
        self.stoped = False
        self.mUtil = mUtil
        self.mutex = QMutex()

    def close_keras_sesion(self):
        print('close_keras_sesion...')
        from keras import backend as K
        K.clear_session()
        self.exit(0)
        pass

    def run(self):
        with QMutexLocker(self.mutex):
            print('in run method,start to send  signal!')
            self.stoped = False

        from train_gesture_model import TrainGestureModeByCNN
        train_net_obj = TrainGestureModeByCNN()
        train_net_obj.trainModel(train_set_path='train_set2', weight_name='myNewCnnModel.hdf5')
        #train_net_obj.trainModel(train_set_path='train_set', weight_name='myNewCnnModel.hdf5')
        self.sendlog.emit(' ')


# 静态图像采集线程
class StaticVideoCollectionThread(QThread):
    sendlog = pyqtSignal(object)
    sendlog_recog = pyqtSignal(object)

    # 使用二值化模型
    def binaryMask(self, frame, x0, y0, width, height):
        # print('use binaryMask model ...')
        minValue = 70
        font = cv2.FONT_HERSHEY_DUPLEX
        # 创建矩形框
        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (255, 255, 0), 1)
        cv2.rectangle(frame, (x0, y0 - 35), (x0 + width, y0), (255, 255, 0), 1)
        cv2.putText(frame, self.guess, (x0 + 60, y0 - 6), font, 1.0, (255, 255, 255), 1)
        roi = frame[y0:y0 + height, x0:x0 + width]
        # 获取灰度图像
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # 高斯模糊:高斯滤波器中像素的权重与其距中心像素的距离成比例
        blur = cv2.GaussianBlur(gray, (5, 5), 2)

        # 图像的二值化提取目标,动态自适应的调整属于自己像素点的阈值，而不是整幅图像都用一个阈值
        th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return res

    # 使用肤色检测模型
    def skinMask(self, frame, x0, y0, width, height):
        print('use skin model ...')
        font = cv2.FONT_HERSHEY_DUPLEX
        # HSV values
        low_range = np.array([0, 50, 80])
        upper_range = np.array([30, 200, 255])

        cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (255, 255, 0), 1)
        cv2.rectangle(frame, (x0, y0 + height - 35), (x0 + width, y0 + height), (255, 255, 0), 1)
        cv2.putText(frame, self.guess, (x0 + 60, y0 + height - 6), font, 1.0, (255, 255, 255), 1)
        roi = frame[y0:y0 + height, x0:x0 + width]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 设阈值，去除背景部分,低于这个lower_red的值，图像值变为0,高于这个upper_red的值，图像值变为0
        mask = cv2.inRange(hsv, low_range, upper_range)
		
		#腐蚀操作，减少整幅图像的白色区域
        mask = cv2.erode(mask, skinkernel, iterations=1)
		#膨胀操作，增加图像中的白色区域
        mask = cv2.dilate(mask, skinkernel, iterations=1)

        # 用高斯分布权值矩阵与原始图像矩阵做卷积运算
        mask = cv2.GaussianBlur(mask, (15, 15), 1)
        # cv2.imshow("Blur", mask)

        # 图像与运算
        res = cv2.bitwise_and(roi, roi, mask=mask)
        # color to grayscale
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        return res

    def __init__(self, video_recongize_obj, parent=None):
        super(StaticVideoCollectionThread, self).__init__(parent)
        self.stoped = False
        self.mutex = QMutex()
        self.cap = None
        self.guess = 'unknown'
        self.mm = mmap.mmap(fileno=-1, length=1024 * 1024, access=mmap.ACCESS_WRITE, tagname='share_mmap')
        self.video_recongize_obj = video_recongize_obj

    def run(self):
        with QMutexLocker(self.mutex):
           print('in run method,start to send  signal!')
           self.stoped = False

        camera.catch_and_recognize_face(self)
        self.cap.release()


    # def run0(self):
    #     with QMutexLocker(self.mutex):
    #         print('in run method,start to send  signal!')
    #         self.stoped = False
    #     # if self.cap == None :
    #     self.cap = cv2.VideoCapture(0)
    #     ret = self.cap.set(3, 640)
    #     ret = self.cap.set(4, 480)
    #     while True:
    #         if self.stoped:
    #           break
    #
    #         ret, frame = self.cap.read()
    #         max_area = 0
    #         frame = cv2.flip(frame, 3)
    #
    #         # 使用面部检测
    #         faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    #
    #         # 转为灰度图像
    #         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
    #
    #         # 画矩形框
    #         for (x, y, w, h) in faces:
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #         # 显示视频
    #         #cv2.imshow('Video', frame)
    #
    #         x0 = 400
    #         y0 = 200
    #         height = 200
    #         width = 200
    #
    #         # 使用肤色检测模型
    #         if self.video_recongize_obj.flag_enable_skin_extraction_model:
    #             roi_frame = self.skinMask(frame, x0, y0, width, height)
    #         else:
    #             # 使用高斯二值化模型
    #             roi_frame = self.binaryMask(frame, x0, y0, width, height)
    #
    #         fram_dict = {}
    #         fram_dict['frame'] = frame
    #         fram_dict['roi_frame'] = roi_frame
    #
    #         self.mm.seek(0)
    #         frame_size = len(frame.tostring())
    #         roi_frame_size = len(roi_frame.tostring())
    #
    #         self.mm.write(frame.tostring())
    #         self.mm.write(roi_frame.tostring())
    #
    #         self.sendlog.emit(str(frame_size))
    #         self.sendlog_recog.emit(str(roi_frame_size))
    #
    #         # print('start to send  signal!')
    #         time.sleep(0.3)
    #     self.cap.release()

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):
        with QMutexLocker(self.mutex):
            return self.stoped


# 批量处理静态图片识别线程
class BatchProcessImgsThread(QThread):
    sendlog = pyqtSignal(object)

    def __init__(self, pic_recog_obj, mUtil, parent=None):
        super(BatchProcessImgsThread, self).__init__(parent)
        self.stoped = False
        self.mutex = QMutex()
        self.pic_recog_obj = pic_recog_obj
        self.mUtil = mUtil

    def run(self):
        with QMutexLocker(self.mutex):
            print('in BatchProcessImgsThread run method,start to send  signal!')
            self.stoped = False

        if self.mUtil.mGesture is None:
            self.mUtil.gestureLoadNN()

        self.pic_recog_obj.getImgListPath(set_path=self.pic_recog_obj.img_test_dir,
                                          imlist=self.pic_recog_obj.png_file_list)
        random.shuffle(self.pic_recog_obj.png_file_list)  # 打乱list
        if len(self.pic_recog_obj.png_file_list) <= 0:
            print('please add imgs for batch recognizaiton!')
            return
        imgs_count = len(self.pic_recog_obj.png_file_list)
        for index, cur_imagName in enumerate(self.pic_recog_obj.png_file_list):
            if self.stoped:
                return
            # imgPath = os.path.join("imgs", cur_imagName)
            imgPath = cur_imagName

            img_dict = {}
            img_dict['imgPath'] = imgPath
            img_dict['number'] = index
            img_dict['imgs_count'] = imgs_count

            self.sendlog.emit(img_dict)
            time.sleep(0.6)

    def stop(self):
        with QMutexLocker(self.mutex):
            self.stoped = True

    def isStoped(self):
        with QMutexLocker(self.mutex):
            return self.stoped


# 手势识别线程
class GestureRecognizeThread(QThread):
    sendlog = pyqtSignal(object)
    sendlog_sample = pyqtSignal(object)

    def __init__(self, mUtil, parent=None):
        super(GestureRecognizeThread, self).__init__(parent)
        self.stoped = False
        self.mutex = QMutex()
        self.mUtil = mUtil
        self.sample_enable = False
        self.counter = 0

    def run(self):
        with QMutexLocker(self.mutex):
            print('in run method,start to send  signal!')
            self.stoped = False

        self.mUtil.gestureLoadNN()
        while True:
            if self.stoped:
                break
            # print ('stoped status is ', self.stoped)
            mm = self.mUtil.getMmap()
            mm.seek(921600)

            roi_frame_read = np.array(np.frombuffer(mm.read(40000), dtype=np.uint8))
            roi_frame_read = roi_frame_read.reshape(200, 200)
            if self.sample_enable:
                data = {}
                data['img'] = roi_frame_read
                data['counter'] = self.counter
                self.counter += 1
                if self.counter > sample_counts:
                    self.counter = 0
                    self.sample_enable = False
                self.sendlog_sample.emit(data)
                time.sleep(0.1)
            else:
                # print ("self.mUtil.mGesture-",self.mUtil.mGesture)
                from datetime import datetime
                a = datetime.now()
                guess, prob = self.mUtil.mGesture.static_video_recognize(roi_frame_read)
                b = datetime.now()
                data = {}
                data['guess'] = guess
                data['prob'] = prob
                data['elapse']= b-a
                self.sendlog.emit(data)
                # 打印手势和识别概率
                print(guess, prob, b-a)
                time.sleep(1)

    def stop_sample_enable(self):
        with QMutexLocker(self.mutex):
            self.sample_enable = False

    def start_sample_enable(self):
        with QMutexLocker(self.mutex):
            self.sample_enable = True
            print('start_sample_enable...')

    def stop(self):
        print('stop ....GestureRecognizeThread before...')
        with QMutexLocker(self.mutex):
            self.stoped = True
            print('stop ....GestureRecognizeThread after ...')

    def isStoped(self):
        with QMutexLocker(self.mutex):
            return self.stoped


# 导入CNN网络线程
class ImportLocalCNN_Net_Thread(QThread):
    sendlog = pyqtSignal(object)

    def __init__(self, video_recognize_obj, parent=None):
        super(ImportLocalCNN_Net_Thread, self).__init__(parent)
        self.video_recognize_obj = video_recognize_obj

    def run(self):
        local_cnn_name,ext = self.video_recognize_obj.local_cnn_name
        self.video_recognize_obj.mUtil.mGesture = None
        self.video_recognize_obj.mUtil.gestureLoadNNByBtn(weight_index=-1, weight_name=local_cnn_name)
        time.sleep(2)
        self.sendlog.emit('网络导入成功！')


# 视频识别widget
class VideoRecognizeWidget(QMainWindow):
    closed_keras_session = pyqtSignal()

    def export_cnn_model_framework(self):
        self.mUtil.mGesture.export_cnn_net_framework()

    def finished_import_local_cnn(self, data):
        QMessageBox.information(self, "通知", "%s" % (data))
        # 开启预测线程
        self.timer.start()
        self.predictThread.start()

    def import_cnn_model(self):
        # 停止预测
        self.predictThread.stop()
        self.timer.stop()
        print('停止预测 ...', self.predictThread)
        time.sleep(0.2)

        local_net_name = QFileDialog.getOpenFileName(self, "open files", '.', "*.hdf5;;*.jpg;;All Files (*)")
        self.local_cnn_name = local_net_name
        # print ('local_net_name='+local_net_name)
        self.cnn_import_thread.start()


    def saveROIImg(self, data):
        img = data['img']
        counter = data['counter']
        print('saveROIImg ....' + str(counter))
        self.progressDialog.setLabelText(self.tr("processing %d/%d" % (counter, sample_counts)))
        self.progressDialog.setValue(counter)
        if self.progressDialog.wasCanceled() or counter >= sample_counts:
            self.predictThread.stop_sample_enable()
            self.btn_static_video_recog.setEnabled(True)
            return

        img_path = 'train_set2/' + self.gestureSampleName + '/' + self.gestureSampleName + '_' + str(counter) + '.png'
        cv2.imwrite(img_path, img)
        time.sleep(0.04)

    def set_hand_extraction_model(self):
        if not self.flag_enable_skin_extraction_model:
            self.btn_set_hand_extraction_model.setText('使用二值化模型')
            self.flag_enable_skin_extraction_model = True
        else:
            self.btn_set_hand_extraction_model.setText('使用肤色模型')
            self.flag_enable_skin_extraction_model = False

        # 使用抽帧的方法录制

    def get_train_sample(self):
        # QPrintDialog
        self.btn_static_video_recog.setDisabled(True)
        gestureName, ok = QInputDialog.getText(self, '手势训练集', '请输入手势名称：')
        if ok:
            print('get dialog :' + gestureName)
            self.gestureSampleName = gestureName
        sample_path = 'train_set2/' + gestureName
        if not os.path.exists(sample_path):
            os.makedirs(sample_path)

        self.progressDialog = QProgressDialog(self)
        self.progressDialog.setWindowModality(Qt.WindowModal)
        self.progressDialog.setMinimumDuration(5)
        self.progressDialog.setWindowTitle(self.tr("Making Training Samples"))
        label_text = "processing %d / %d" % (0, sample_counts)
        self.progressDialog.setLabelText(self.tr(label_text))
        # self.progressDialog.setLabelText(self.tr("Copy......"))
        self.progressDialog.setCancelButtonText(self.tr("Cancel"))
        self.progressDialog.setRange(0, sample_counts)
        self.progressDialog.setMinimumWidth(500)
        self.progressDialog.move(200, 200)
        self.predictThread.start_sample_enable()

    def __init__(self, mUtil, parent=None):
        super(VideoRecognizeWidget, self).__init__(parent)
        self.setStyleSheet("background: rgb(148, 219, 95)")
        self.png_file_list = []
        # self.setStyleSheet("background: red")
        self.curImgName = ''
        self.mUtil = mUtil
        self.gestureSampleName = 'none'
        self.progressDialog = None

        widget = QWidget()
        vbox = QHBoxLayout()

        # 设置 图片显示区域
        # 设置 图片显示区域
        groupBox_img_display = QGroupBox(u"视频显示区域")
        vbox.addWidget(groupBox_img_display)
        self.vbox_png = QVBoxLayout()
        groupBox_img_display.setLayout(self.vbox_png)
        self.label_img = QLabel('')
        self.image = QImage()

        self.label_img.setPixmap(QPixmap('ico/gesture.png'))
        self.vbox_png.addWidget(self.label_img)
        self.label_img.setMinimumWidth(self.width() * 4 / 5)

        subHvbox = QVBoxLayout()
        groupBox_gesture_preview = QGroupBox(u"手势预览")
        subHvbox.addWidget(groupBox_gesture_preview)
        self.vbox_roi_layout = QVBoxLayout()
        self.label_roi = QLabel('')

        self.vbox_roi_layout.addWidget(self.label_roi)
        groupBox_gesture_preview.setLayout(self.vbox_roi_layout)
        groupBox_gesture_preview.setMinimumHeight(240)

        groupBox_result_display = QGroupBox(u"结果显示")
        subHvbox.addWidget(groupBox_result_display)
        self.vbox_result = QGridLayout()
        groupBox_result_display.setLayout(self.vbox_result)

        self.lable_result_title = QLabel('预测值')
        self.lable_result = QTextEdit()
        self.lable_proba_title = QLabel('准确率')
        self.lable_proba = QTextEdit()
        self.lable_result.setMaximumHeight(30)
        self.lable_proba.setMaximumHeight(30)
        groupBox_result_display.setMaximumWidth(self.width() / 3)

        self.vbox_result.addWidget(self.lable_result_title, 0, 0)
        self.vbox_result.addWidget(self.lable_result, 0, 1)
        self.vbox_result.addWidget(self.lable_proba_title, 1, 0)
        self.vbox_result.addWidget(self.lable_proba, 1, 1)

        # vbox.addStretch(1)
        groupBox_function = QGroupBox(u"功能")
        # vbox.addWidget(groupBox_function)
        subHvbox.addWidget(groupBox_function)
        self.vbox_btn = QVBoxLayout()
        groupBox_function.setLayout(self.vbox_btn)
        vbox.addLayout(subHvbox)

        self.btn_set_hand_extraction_model = QPushButton(u"使用肤色模型")
        self.btn_static_video_recog = QPushButton(u"制作样本")
        self.btn_train_gesture_sample = QPushButton(u"训练样本")
        self.btn_import_cnn_model = QPushButton(u"导入网络")
        self.btn_export_cnn_model = QPushButton(u"导出网络结构")

        # 设置伸缩参数
        self.vbox_btn.addStretch(10)
        self.vbox_btn.addWidget(self.btn_set_hand_extraction_model)
        self.vbox_btn.addWidget(self.btn_static_video_recog)
        self.vbox_btn.addWidget(self.btn_train_gesture_sample)
        self.vbox_btn.addWidget(self.btn_import_cnn_model)
        self.vbox_btn.addWidget(self.btn_export_cnn_model)
        # 设置占用比例
        groupBox_function.setMaximumWidth(self.width() / 3)

        # 加载初始页面
        if self.image.load("ico/love.jpg"):
            self.label_img.setPixmap(QPixmap.fromImage(self.image))

        # 设置窗口中心显示
        # self.btn_capture_img.setGeometry(10, 10, 30, 30)

        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        self.setWindowTitle(u'img display')

        self.flag_enable_skin_extraction_model = False
        self.timer = StaticVideoCollectionThread(self)

        self.cnn_import_thread = ImportLocalCNN_Net_Thread(self)
        self.cnn_import_thread.sendlog.connect(self.finished_import_local_cnn)

        self.timer.sendlog.connect(self.update_image)
        #初始化CNN模型
        self.predictThread = GestureRecognizeThread(mUtil)
        self.predictThread.sendlog.connect(self.static_video_hand_gesture_recognize)
        self.predictThread.sendlog_sample.connect(self.saveROIImg)
        self.predictThread.sendlog.connect(self.setTimer)

        self.btn_static_video_recog.clicked.connect(self.get_train_sample)
        self.btn_set_hand_extraction_model.clicked.connect(self.set_hand_extraction_model)
        self.btn_import_cnn_model.clicked.connect(self.import_cnn_model)
        self.btn_export_cnn_model.clicked.connect(self.export_cnn_model_framework)

        self.train_cnn = TrainCNNThread(mUtil)
        self.train_cnn.sendlog.connect(self.train_cnn_net)

        self.btn_train_gesture_sample.clicked.connect(self.train_cnn_net)
        self.is_traning_flag = False
        # self.predictThread.start()
        self.closed_keras_session.connect(self.train_cnn.close_keras_sesion)  # self.show()

        # 设置本地CNN 名
        self.local_cnn_name = ''

    def train_cnn_net(self):
        if self.is_traning_flag:
            self.btn_train_gesture_sample.setText('训练样本')
            self.closed_keras_session.emit()
            time.sleep(0.2)
            self.train_cnn.quit()
            time.sleep(0.2)

            self.is_traning_flag = False
            self.timer.start()
            self.predictThread.start()
            return
        else:
            self.btn_train_gesture_sample.setText('停止训练')
            self.is_traning_flag = True
            self.mUtil.mGesture = None  # 清空手势keras

        self.timer.stop()
        self.predictThread.stop()
        time.sleep(1)
        self.train_cnn.start()

    def static_video_hand_gesture_recognize(self, data):
        guess = data['guess']
        prob = data['prob']
        self.lable_result.setText(guess)
        self.lable_proba.setText('%.2f%%' % prob)

    def setTimer(self,data):
        guess = data['guess']
        prob = data['prob']
        self.timer.guess = guess

    # 在UI界面中更新视频图像和ROI预览图像
    def update_image(self, frame_size):
        mm = self.mUtil.getMmap()
        # 从共享内存中读取图像
        mm.seek(0)
        frame_read = np.array(np.frombuffer(mm.read(921600), dtype=np.uint8))
        frame_read = frame_read.reshape(480, 640, 3)

        # mm.seek(0)
        #roi_frame_read = np.array(np.fromstring(mm.read(40000), dtype=np.uint8))
        roi_frame_read = np.array(np.frombuffer(mm.read(40000), dtype=np.uint8))
        roi_frame_read = roi_frame_read.reshape(200, 200)

        frame_read = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        self.image = qimage2ndarray.array2qimage(frame_read, normalize=True)

        # self.roi_frame = data['roi_frame']
        roi_image = qimage2ndarray.array2qimage(roi_frame_read, normalize=True)

        self.label_img.setPixmap(QPixmap.fromImage(self.image))  # 一帧一帧的显示
        self.label_roi.setPixmap(QPixmap.fromImage(roi_image))

# 图片识别widget
class ImagesRecognizeWidget(QMainWindow):
    def hand_gesture_recognize(self):
        if self.curImgName:
            guess = self.mUtil.mGesture.static_image_recognize(self.curImgName)
            self.lable_result.setText(guess)
            print('src:{}---recognizaiton:{}'.format(self.curImgName, guess))
        else:
            print('please select img file!')

    def batch_update_image(self, img_dict):
        imgPath = img_dict['imgPath']
        index = img_dict['number']
        imgs_count = img_dict['imgs_count']
        guess = self.mUtil.mGesture.static_image_recognize(imgPath)
        label = guess.lower()

        self.lable_result.setText(guess)
        self.label_png.setPixmap(QPixmap(imgPath))
        print(imgPath)

        self.predict_count += 1
        if label in imgPath:
            self.predict_ok_count += 1
            self.lable_result_status.setPixmap(QPixmap('ico/ok.png'))
        else:
            self.lable_result_status.setPixmap(QPixmap('ico/fail.png'))
        self.lable_proba.setText('%.2f%%' % (100 * self.predict_ok_count / self.predict_count))
        self.lable_imgCount.setText(str(index + 1) + '/' + str(imgs_count))

    def batch_hand_gesture_recognize(self):
        if not self.flag_touch_btn_batchCapture:
            self.flag_touch_btn_batchCapture = True
            self.btn_batchCapture_img.setText('停止')
            self.cap_timer.start()
            self.predict_count = 0
            self.predict_ok_count = 0
        else:
            self.flag_touch_btn_batchCapture = False
            self.btn_batchCapture_img.setText('批量识别')
            self.cap_timer.stop()

    def getImgListPath(self, set_path, imlist):
        all_path = os.listdir(set_path)
        for f in all_path:
            p = os.path.abspath(os.path.join(set_path, f))
            if os.path.isdir(p):
                self.getImgListPath(p, imlist)
            elif os.path.isfile(p):
                if os.path.splitext(p)[1] == '.png' and p not in imlist:
                    imlist.append(p)

    def open_prev_img(self):
        self.getImgListPath(set_path=self.img_test_dir, imlist=self.png_file_list)
        if len(self.png_file_list) == 0:
            print('please add imgs!')
            return
        if self.curImgName:
            curPngIndex = self.png_file_list.index(self.curImgName)
            curPngIndex -= 1
            if curPngIndex <= -1:
                curPngIndex = len(self.png_file_list) - 1
            imgName = self.png_file_list[curPngIndex]
            # print('prev img ' + imgName)
            self.curImgName = imgName
            self.label_png.setPixmap(QPixmap(self.curImgName))
        else:
            print('please select img file!')

    def open_next_img(self):
        # print (self.img_test_dir)
        self.getImgListPath(set_path=self.img_test_dir, imlist=self.png_file_list)
        if len(self.png_file_list) == 0:
            print('please add imgs!')
            return
        if self.curImgName:
            # print (self.png_file_list)
            curPngIndex = self.png_file_list.index(self.curImgName)
            curPngIndex += 1
            if curPngIndex >= len(self.png_file_list):
                curPngIndex = 0
            imgName = self.png_file_list[curPngIndex]
            # print('next img ' + imgName)
            self.curImgName = imgName
            self.label_png.setPixmap(QPixmap(imgName))
        else:
            print('please select img file!')

    def open_img(self):
        imgName,type = QFileDialog.getOpenFileName(self, "open files", self.img_test_dir, "*.png;;*.jpg;;All Files (*)")
        self.img_test_dir = os.path.dirname(imgName)
        print(self.img_test_dir)
        # 转换路径(linux->windows)
        self.curImgName = re.compile('\/').sub('\\\\', imgName)
        self.label_png.setPixmap(QPixmap(imgName))

    def display_layer_output(self):
        net_layer_index, ok = QInputDialog.getText(self, '手势网络层显示', '网络层数：')
        if ok:
            print('get dialog :' + net_layer_index)
        if self.curImgName:
            self.mUtil.mGesture.visualizeLayers(imgName=self.curImgName, layerIndex=int(net_layer_index))
        else:
            self.mUtil.mGesture.visualizeLayers(imgName='train_set/eight/eight_25.png', layerIndex=int(net_layer_index))

        pass

    def __init__(self, mUtil, parent=None):
        super(ImagesRecognizeWidget, self).__init__(parent)
        self.png_file_list = []
        self.setStyleSheet("background: rgb(85, 170, 255)")
        self.curImgName = ''
        # 本地测试图片文件夹
        self.img_test_dir = "imgs_test"
        self.mUtil = mUtil

        widget = QWidget()
        vbox = QHBoxLayout()

        # 设置 图片显示区域
        groupBox_img_display = QGroupBox(u"图片显示区域")
        vbox.addWidget(groupBox_img_display)
        self.vbox_png = QVBoxLayout()
        groupBox_img_display.setLayout(self.vbox_png)
        self.label_png = QLabel(self)

        self.label_png.setPixmap(QPixmap('ico/title.jpeg'))
        self.vbox_png.addWidget(self.label_png)
        groupBox_img_display.setMinimumWidth(self.width() / 2)
        self.vbox_png.addStretch(1)

        subHvbox = QVBoxLayout()
        groupBox_model = QGroupBox(u"模式")
        subHvbox.addWidget(groupBox_model)
        self.vbox_radio_layout = QVBoxLayout()
        self.lable_radio_debug = QRadioButton('debug', self)
        self.lable_radio_release = QRadioButton('release', self)
        self.lable_radio_release.setFocusPolicy(Qt.NoFocus)
        self.lable_radio_debug.toggled.connect(self.mUtil.gestureLoadNN)
        self.lable_radio_debug.toggle()  # 显示选中，可以用来加载默认的神经网络
        # self.lable_radio_release.toggle()  # 默认显示选中

        self.vbox_radio_layout.addWidget(self.lable_radio_debug)
        self.vbox_radio_layout.addWidget(self.lable_radio_release)
        groupBox_model.setLayout(self.vbox_radio_layout)

        groupBox_result_display = QGroupBox(u"结果显示")
        subHvbox.addWidget(groupBox_result_display)
        self.vbox_result = QGridLayout()
        self.lable_result_title = QLabel('预测值')
        self.lable_result = QTextEdit()
        self.lable_proba_title = QLabel('准确率')
        self.lable_proba = QTextEdit()
        self.lable_imgCount_title = QLabel('图片张数')
        self.lable_imgCount = QTextEdit()
        self.lable_imgCount.setMinimumWidth(20)
        self.lable_status_title = QLabel('预测\n状态')
        self.lable_result_status = QLabel('')
        self.lable_result.setFocusPolicy(False)
        self.vbox_result.addWidget(self.lable_result_title, 0, 0)
        self.vbox_result.addWidget(self.lable_result, 0, 1)
        self.vbox_result.addWidget(self.lable_proba_title, 1, 0)
        self.vbox_result.addWidget(self.lable_proba, 1, 1)
        self.vbox_result.addWidget(self.lable_imgCount_title, 2, 0)
        self.vbox_result.addWidget(self.lable_imgCount, 2, 1)

        self.vbox_result.addWidget(self.lable_status_title, 3, 0)
        self.vbox_result.addWidget(self.lable_result_status, 3, 1)
        groupBox_result_display.setMaximumWidth(self.width() / 3)

        groupBox_result_display.setLayout(self.vbox_result)

        groupBox_img_operation = QGroupBox(u"图片操作")
        subHvbox.addWidget(groupBox_img_operation)
        self.vbox_btn = QVBoxLayout()
        groupBox_img_operation.setLayout(self.vbox_btn)
        groupBox_img_operation.setAlignment(Qt.AlignTop)
        vbox.addLayout(subHvbox)
        # QMargins定义了矩形的四个外边距量，left, top, right和bottom，描述围绕矩形的边框宽度

        self.btn_capture_img = QPushButton(u"打开")
        self.btn_train_gesture_sample = QPushButton(u"上一个")
        self.btn_import_cnn_model = QPushButton(u"下一个")
        self.btn_export_cnn_model = QPushButton(u"识别")
        self.btn_display_layer_output = QPushButton(u"显示网络输出")
        self.btn_batchCapture_img = QPushButton(u"批量识别")

        # 设置伸缩参数
        self.vbox_btn.addWidget(self.btn_capture_img)
        self.vbox_btn.addWidget(self.btn_train_gesture_sample)
        self.vbox_btn.addWidget(self.btn_import_cnn_model)
        self.vbox_btn.addWidget(self.btn_export_cnn_model)
        self.vbox_btn.addWidget(self.btn_display_layer_output)
        self.vbox_btn.addWidget(self.btn_batchCapture_img)

        # 设置占用比例
        groupBox_img_operation.setMaximumWidth(self.width() / 3)

        self.btn_capture_img.clicked.connect(self.open_img)
        self.btn_train_gesture_sample.clicked.connect(self.open_prev_img)
        self.btn_import_cnn_model.clicked.connect(self.open_next_img)
        self.btn_export_cnn_model.clicked.connect(self.hand_gesture_recognize)
        self.btn_batchCapture_img.clicked.connect(self.batch_hand_gesture_recognize)
        self.btn_display_layer_output.clicked.connect(self.display_layer_output)
        self.flag_touch_btn_batchCapture = False
        # 设置窗口中心显示
        # self.btn_capture_img.setGeometry(10, 10, 30, 30)

        widget.setLayout(vbox)
        self.setCentralWidget(widget)

        self.setWindowTitle(u'img display')
        self.cap_timer = BatchProcessImgsThread(self, self.mUtil)
        self.cap_timer.sendlog.connect(self.batch_update_image)  # self.show()
        self.predict_count = 0
        self.predict_ok_count = 0


# 主界面主要包含两个widget,图片的识别和视频的识别
class MyQWidget(QWidget):
    def __init__(self, parent=None):
        super(MyQWidget, self).__init__(parent)
        self.tabwidget = QTabWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabwidget)
        # 允许tab点击关闭
        self.tabwidget.setTabsClosable(True)
        self.tabwidget.setFixedSize(1000, 600)
        mUtil = UtilOperation()
        self.captureRecog = ImagesRecognizeWidget(mUtil)
        self.videoRecog = VideoRecognizeWidget(mUtil)
        self.tabwidget.addTab(self.captureRecog, u"静态图片识别")
        self.tabwidget.addTab(self.videoRecog, u"静态视频识别")
        #self.tabwidget.connect(self.tabwidget, signal("currentChanged(int)"), self.tabChangedSlot)
        self.tabwidget.currentChanged.connect(self.tabChangedSlot)

        # 设置最小化、最大化窗口
        self.setWindowFlags(Qt.Window)
        # 设置图标
        self.setWindowIcon(QIcon('ico/gesture.png'))

        self.setWindowTitle('手势识别系统')
        self.show()

    def tabChangedSlot(self, argTabIndex):
        print(argTabIndex)

        if (argTabIndex == 1):
            self.videoRecog.timer.start()
            self.videoRecog.predictThread.start()
        else:
            self.videoRecog.timer.stop()
            self.videoRecog.predictThread.stop()

if __name__ == '__main__':
    font = QFont("黑体", 12)
    QApplication.setFont(font)
    a = QApplication(sys.argv)
    m = MyQWidget()
    m.show()
    sys.exit(a.exec_())
