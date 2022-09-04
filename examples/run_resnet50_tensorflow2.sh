#!/bin/bash -f
source ./settings.sh
cd ./examples
./run_classification.sh -n resnet50 -f tensorflow2
#./run_classification.sh -n resnet50 -f tensorflow2 2>&1 > log/u50lv_resent50_tensorflow2.log

