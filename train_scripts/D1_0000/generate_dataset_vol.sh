#!/bin/bash
#OAR -p cluster='graffiti'
#OAR -l walltime=15:00:00
#OAR -O /home/psrivastava/logs_oarsub/oar_job.%jobid%.output
#OAR -E /home/psrivastava/logs_oarsub/oar_job.%jobid%.error
set -xv 

/home/psrivastava/axis-2/IWAENC/train_scripts/D1_0000/run_cmd.sh




