#!/bin/bash
set -ex
LD_LIBRARY_PATH=/usr/local/cuda/lib/ ../../darknet/darknet detector test cfg-voc.data yolov3-tiny-trafficlights-test.cfg backup/yolov3-tiny-trafficlights-train_500.weights red.jpg -thresh 0
#for i in just_traffic_lights/*.jpg; do
#  ../../darknet/darknet detector test cfg-voc.data yolov3-tiny-trafficlights-test.cfg backup/yolov3-tiny-trafficlights-train_500.weights $i -thresh 0
#done
