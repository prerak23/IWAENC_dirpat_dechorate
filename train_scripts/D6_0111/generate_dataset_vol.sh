#!/bin/bash
#OAR -p cluster='graffiti'
#OAR -l walltime=18:00:00
#OAR -O /home/psrivastava/logs_oarsub/D6/oar_job.%jobid%.output
#OAR -E /home/psrivastava/logs_oarsub/D6/oar_job.%jobid%.error
set -xv 

/home/psrivastava/axis-2/IWAENC/train_scripts/D6_0111/run_cmd.sh




