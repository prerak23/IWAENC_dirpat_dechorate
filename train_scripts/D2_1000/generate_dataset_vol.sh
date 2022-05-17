#!/bin/bash
#OAR -p cluster='graphique'
#OAR -l walltime=15:00:00
#OAR -O /home/psrivastava/logs_oarsub/training/D2/oar_job.%jobid%.output
#OAR -E /home/psrivastava/logs_oarsub/training/D2/oar_job.%jobid%.error
set -xv 

/home/psrivastava/axis-2/IWAENC/train_scripts/D2_1000/run_cmd.sh




