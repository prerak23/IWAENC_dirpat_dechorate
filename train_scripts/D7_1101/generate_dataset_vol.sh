#!/bin/bash
#OAR -p cluster='grele'
#OAR -l walltime=18:00:00
#OAR -O /home/psrivastava/logs_oarsub/D7/oar_job.%jobid%.output
#OAR -E /home/psrivastava/logs_oarsub/D7/oar_job.%jobid%.error
set -xv 

/home/psrivastava/axis-2/IWAENC/train_scripts/D7_1101/run_cmd.sh




