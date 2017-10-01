## Vehicle Detection with CNNs

These presentation slides and notebook are meant to give a basic overview of basic techniques in computer vision, relating to classification, localization and detection.

### Classfication and Localization (Part 1)

Part 1 covers how to setup and train a CNN (LeNet architecture) using Tensorflow core. Notebook [here](https://github.com/marc-chan/traffic/blob/master/vehicle_localization_demo.ipynb).

<img src="https://raw.githubusercontent.com/marc-chan/traffic/master/img/localization.png" width="300" height="300">

### Detection (Part 2)

Part 2 covers how to use a pretrained network (tiny YOLO v1) to perform inference in Keras 2. Notebook [here](https://github.com/marc-chan/traffic/blob/master/vehicle_detection_demo.ipynb).

<img src="https://raw.githubusercontent.com/marc-chan/traffic/master/img/sample.png" width="300" height="300">

### Downloads

Image data (Part 1) and pretrained weight files (Part 2) are too big to be hosted on Github. Download them from the links below:

* [Image Data](https://www.dropbox.com/s/qg3x8ulzs4t1ks8/working_set.p)
* [Weights (Numpy)](https://www.dropbox.com/s/xpqm8vjzpidjeqo/yolo_tiny_weights.npy)
* [Weights (h5)](https://www.dropbox.com/s/4l82xgalhekfm1x/yolo_tiny.h5)
