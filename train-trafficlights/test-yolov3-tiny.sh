#!/bin/bash
set -ex
../../darknet/darknet detector test cfg-voc.data yolov3-tiny-trafficlights-test.cfg backup/yolov3-tiny-trafficlights-train_500.weights red.jpg -thresh 0
