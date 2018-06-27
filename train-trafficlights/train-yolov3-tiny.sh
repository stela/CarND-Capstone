#!/bin/bash
set -ex
../../darknet/darknet detector train cfg-voc.data yolov3-tiny-trafficlights-train.cfg ../../darknet/darknet53.conv.74
#  -gpus 0
