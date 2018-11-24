#!/bin/bash 
SRC_MODEL=$1
INF_MODEL=$2

python ./keras-retinanet/keras_retinanet/bin/convert_model.py $SRC_MODEL $INF_MODEL