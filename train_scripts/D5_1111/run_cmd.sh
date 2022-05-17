#!/bin/bash 

source /home/psrivastava/new_env/bin/activate
export PYTHONPATH=/home/psrivastava/pyroomacoustics/:$PYTHONPATH
python /home/psrivastava/axis-2/IWAENC/train_scripts/D5_1111/mlh_rt60.py
