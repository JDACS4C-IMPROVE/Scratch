#!/bin/zsh -f

# JOB SH

# Set up modules:
source /usr/share/lmod/lmod/init/zsh
MODULEPATH+=:/opt/cray/pals/lmod/modulefiles/core
MODULEPATH+=:/opt/aurora/24.347.0/modulefiles
export MODULEPATH
module load cray-pals frameworks

set -eux

# Unpack and use a Python environment
# (a customized clone of module frameworks):
nice tar xf /home/wozniak/Public/sfw/aurora/TF.tar -C /tmp
PATH=/tmp/TF/bin:$PATH

# The application locations:
SFW=/home/wozniak/Public/sfw
UNO=$SFW/I-UNO.jw
IMPROVE=$SFW/IMPROVE
export PYTHONPATH=$IMPROVE:${PYTHONPATH:-}

# Application arguments:
A=( --input_dir  $UNO/exp_result
    --output_dir /tmp/train-out
    --epochs 1
  )

# Train the model!
mpiexec python $UNO/uno_train_improve.py $A
