#!/bin/bash
#OAR -p cluster='graphique'
#OAR -l walltime=15:00:00
#OAR -O /home/psrivastava/logs_oarsub/training/oar_job.%jobid%.output
#OAR -E /home/psrivastava/logs_oarsub/training/oar_job.%jobid%.error
set -xv 

/home/psrivastava/axis-2/IWAENC/train_scripts/roomsim_train/run_cmd.sh




