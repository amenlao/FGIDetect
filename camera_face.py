# -*- coding: utf-8 -*-
import face_recognition
import cv2
import time
from read_data import read_name_list
from read_data import read_file

#读取摄像头，并识别摄像头中的人脸，进行匹配。
def catch_and_recognize_face(self) :

    all_encoding, lable_list, counter = read_file('./dataset')
    name_list = read_name_list('./dataset')
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    # video_capture = cv2.VideoCapture(0)
    video_capture = cv2.VideoCapture(0)
    ret = video_capture.set(3, 640)
    ret = video_capture.set(4, 480)
    name = "unknown"
    while True:
         ret, frame = video_capture.read()
         # 图像左右翻转
         frame = cv2.flip(frame, 3)
         small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

         if process_this_frame:
             face_locations = face_recognition.face_locations(small_frame)
             face_encodings = face_recognition.face_encodings(small_frame, face_locations)
             face_names = []
             #匹配，并赋值
             for face_encoding in face_encodings:
                 i = 0
                 j = 0
                 for t in all_encoding:
                     for k in t:
                         match = face_recognition.compare_faces([k], face_encoding)
                         if match[0]:
                             name = name_list[i]
                             j=1
                     i = i+1
                 if j == 0:
                     name = "unknown"

                 face_names.append(name)

         process_this_frame = not process_this_frame

         # 显示人脸框和名字
         for (top, right, bottom, left), name in zip(face_locations, face_names):
             top *= 4
             right *= 4
             bottom *= 4
             left *= 4

             cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255),  2)
             cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
             font = cv2.FONT_HERSHEY_DUPLEX
             cv2.putText(frame, name, (left+6, bottom-6), font, 1.0, (255, 255, 255), 1)

         x0 = 400
         y0 = 200
         height = 200
         width = 200

         # 使用肤色检测模型
         if self.video_recongize_obj.flag_enable_skin_extraction_model:
             roi_frame = self.skinMask(frame, x0, y0, width, height)
         else:
             # 使用高斯二值化模型
             roi_frame = self.binaryMask(frame, x0, y0, width, height)

         fram_dict = {}
         fram_dict['frame'] = frame
         fram_dict['roi_frame'] = roi_frame

         self.mm.seek(0)
         frame_size = len(frame.tostring())
         roi_frame_size = len(roi_frame.tostring())

         self.mm.write(frame.tostring())
         self.mm.write(roi_frame.tostring())

         self.sendlog.emit(str(frame_size))
         self.sendlog_recog.emit(str(roi_frame_size))

         #print(name)
         time.sleep(0.3)



         #cv2.imshow('Video', frame)
         #if cv2.waitKey(1) & 0xFF == ord('q'):
         #    break

    #video_capture.release()
    #cv2.destroyAllWindows()

def recognize_test():
    all_encoding, lable_list, counter = read_file('./dataset')
    name_list = read_name_list('./dataset')
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    video_capture = cv2.VideoCapture(0)
    # video_capture = cv2.VideoCapture(0)
    # ret = video_capture.set(3, 640)
    # ret = video_capture.set(4, 480)
    while True:
        ret, frame = video_capture.read()
        # 图像左右翻转
        # frame = cv2.flip(frame, 3)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        if process_this_frame:
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            face_names = []
            # 匹配，并赋值
            for face_encoding in face_encodings:
                i = 0
                j = 0
                for t in all_encoding:
                    for k in t:
                        match = face_recognition.compare_faces([k], face_encoding)
                        if match[0]:
                            name = name_list[i]
                            j = 1
                    i = i + 1
                if j == 0:
                    name = "unknown"

                face_names.append(name)

        process_this_frame = not process_this_frame

        # 显示人脸框和名字
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # video_capture.release()
    # cv2.destroyAllWindows()

#测试self
if __name__ == '__main__':
    recognize_test()
    print("run successful")