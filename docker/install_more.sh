#!/bin/bash

set -euxo pipefail

sudo apt-get update
sudo python3.6 -m pip install requests
sudo python3.6 -m pip install matplotlib
sudo python3.6 -m pip install scipy
sudo python3.6 -m pip install imageio==2.6.0

sudo python3.6 -m pip install scikit-image

sudo python3.6 -m pip install tensorboard_logger \
    tensorflow

# seems that we need this version of tensorboard
# maybe because tensorboard_logger is not compatible 
# with newer versions of tensorboard?
sudo python3.6 -m pip install tensorboard==1.8.0

sudo python3.6 -m pip install sklearn

sudo python3.6 -m pip install opencv-contrib-python


sudo apt install python3.6-tk \
    ffmpeg
