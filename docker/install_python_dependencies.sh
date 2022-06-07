#!/bin/bash

apt-get install software-properties-common
sudo add-apt-repository -y ppa:jblgf0/python
apt-get update
apt install python3.6
wget https://bootstrap.pypa.io/pip/3.6/get-pip.py 
python3.6 get-pip.py
apt-get install -y python3.6-setuptools
#pip install --upgrade pip==9.0.3
#apt-get install python-setuptools
#pip install --upgrade pip
python3.6 -m pip install --upgrade pip
# python3.6 -m pip install -U setuptools



python3.6 -m pip install \
  jupyter \
  opencv-python==4.2.0.32 \
  plyfile \
  pandas \
  tensorflow \
  numpy \
  scipy \
  matplotlib \
  pyyaml \
  future \
  typing \
  tqdm \
  ipdb \
  ipykernel \
  tensorboard_logger


python3.6 -m pip install --upgrade torchvision

apt-get -y install ipython ipython-notebook
python3.6 -m pip ipykernel install --user