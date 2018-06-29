#!/bin/bash
set -ex
#../../darknet/darknet detector train cfg-voc.data yolov3-tiny-trafficlights-train.cfg ../../darknet/darknet53.conv.74
#LD_LIBRARY_PATH=/usr/local/cuda/lib/ ../../darknet/darknet detector train cfg-voc.data yolov3-tiny-trafficlights-train.cfg ../../darknet/yolov3-tiny.weights
LD_LIBRARY_PATH=/usr/local/cuda/lib/ ../../darknet/darknet detector train cfg-voc.data yolov3-tiny-trafficlights-train.cfg backup/yolov3-tiny-trafficlights-train.backup
#../../darknet/darknet detector train cfg-voc.data yolov3-tiny-trafficlights-train.cfg backup/yolov3-tiny-trafficlights-train_200.weights
#  -gpus 0
