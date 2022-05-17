#!/bin/sh
#OAR -l core=2,walltime=20:00:00
#OAR -p gpu_count='0'
#OAR --array-param-file /home/psrivastava/axis-2/IWAENC/z_test/jobs_test.txt
#OAR -O /home/psrivastava/logs_oarsub/D1/oar_job.%jobid%.output
#OAR -E /home/psrivastava/logs_oarsub/D1/oar_job.%jobid%.error
set -xv

/home/psrivastava/axis-2/IWAENC/z_test/D1_test_set/run_cmd.sh $*

# source /home/psrivastava/new_env/bin/activate
# export PYTHONPATH=/home/psrivastava/pyroomacoustics/:$PYTHONPATH
# python /home/psrivastava/axis-2/IWAENC/dataset_generation/D1_0000/create_rirs.py $*
