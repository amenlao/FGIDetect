# FGIDetect
Face and Gesture Image Detection
FGIDetect is a project to detect face and guesture using camera capture technology.

## Requirements
- Python 3.6.1 OpenCV 3.4.1 Keras 2.0.2 Tensorflow 1.2.1 Theano 0.9.0
- Windows or Linux (macOS not officially supported, but might work)
- Suggestion: Better to download Anaconda as it will take care of most of the other packages and easier to setup a virtual workspace to work with multiple versions of key packages like python, opencv etc.
## Repo contents
- ui_setup.py: The main script launcher. This file contains all the code for UI options and OpenCV code to capture camera contents. This script internally calls interfaces to gesture_recognize.py and camera_face.py.
- gesture_recognize.py: This script file holds all the CNN specific code to create CNN model, load the weight file (if model is pretrained), train the model using image samples present in ./train_set2, visualize the feature maps at different layers of NN (of pretrained model) for a given input image present in ./train_set2 folder.
- camera_face.py: This script file holds the face recognition code to recognize face through camera video, load the user  name using face image samples present in ./dataset folder.
## Installation Options:
- install this module from pypi using pip3 (or pip2 for Python 2):
  ```bash
  pip3 install face_recognition
  ```
## Usage
- On Windows 
  ```bash
  python ui_setup.py
  ```
## features
This application comes with CNN model to recognize upto 4 pretrained gestures:
- OK
- One
- Two
- Five

This application provides following functionalities:
- Prediction : Which allows the app to guess the user's gesture against pretrained gestures. App can dump the prediction data to the console terminal or to a json file directly which can be used to plot real time prediction bar chart (you can use my other script
- New Training : Which allows the user to retrain the NN model. User can change the model architecture or add/remove new gestures. This app has inbuilt options to allow the user to create new image samples of user defined gestures if required.
- Visualization : Which allows the user to see feature maps of different NN layers for a given input gesture image. Interesting to see how NN works and learns things.
