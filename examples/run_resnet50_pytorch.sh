#!/bin/bash -f
source ./settings.sh
cd ./examples
./run_classification.sh -n resnet50 -f pytorch
#./run_classification.sh -n resnet50 -f pytorch 2>&1 > log/u50lv_resnet50_pytorch.log

