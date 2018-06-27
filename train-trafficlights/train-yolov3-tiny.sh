#!/bin/bash
set -ex
../../darknet/darknet detector train cfg-voc.data yolov3-tiny-trafficlights-train.cfg
#  -gpus 0
#  ../../darknet/yolov3-tiny.weights
