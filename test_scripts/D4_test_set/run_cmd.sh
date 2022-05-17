#!/bin/bash
echo $1 

source /home/psrivastava/new_env/bin/activate
export PYTHONPATH=/home/psrivastava/pyroomacoustics/:$PYTHONPATH
python /home/psrivastava/axis-2/IWAENC/z_test/D4_test_set/noisy_mixture_final.py $1
