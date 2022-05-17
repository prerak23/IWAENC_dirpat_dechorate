#!/bin/bash
#OAR -p cluster='grele'
#OAR -l walltime=15:00:00
#OAR -O /home/psrivastava/logs_oarsub/training/D3/oar_job.%jobid%.output
#OAR -E /home/psrivastava/logs_oarsub/training/D3/oar_job.%jobid%.error
set -xv 

/home/psrivastava/axis-2/IWAENC/train_scripts/D3_1010/run_cmd.sh




