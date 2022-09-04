#!/bin/bash -f 
source ./settings.sh
./examples/docker/run_ee.sh
#./examples/docker/run.bash

#./examples/docker/run.bash -- 'ls'
#./examples/docker/run.bash -- '. zebra/settings.sh'
#./examples/docker/run.bash -- 'source zebra/settings.sh'
#./examples/docker/run.bash -- '. zebra/settings.sh; zebra_tools --config'
#./examples/docker/run.bash -- '. zebra/settings.sh tensorflow ; zebra_tools --config'

