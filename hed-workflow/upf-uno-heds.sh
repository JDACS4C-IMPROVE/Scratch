#!/bin/bash
set -eu

# UPF UNO HED

if (( ${#} != 1 ))
then
  echo "usage: upf-uno-hed.sh SITE"
  exit 1
fi

SITE=$1

SUPERVISOR_TOOL=$( which supervisor || echo NONE )
if [[ $SUPERVISOR_TOOL == NONE ]]
then
  echo "Put supervisor in PATH!"
  exit 1
fi

# Self-configure
THIS=$(               cd $( dirname $0 ) ; /bin/pwd )
# EMEWS_PROJECT_ROOT=$( cd $THIS/..                    ; /bin/pwd )
SUPERVISOR_HOME=$( cd $( dirname $SUPERVISOR_TOOL )/.. ; /bin/pwd )
WORKFLOWS_ROOT=$(  cd $SUPERVISOR_HOME/workflows ; /bin/pwd )
# export EMEWS_PROJECT_ROOT

source $WORKFLOWS_ROOT/common/sh/utils.sh
sv_path_prepend $WORKFLOWS_ROOT/common/sh

export OBJ_RETURN="val_loss"
CFG_SYS=cfg-sys-1.sh

export CANDLE_FRAMEWORK="keras"
export CANDLE_MODEL_TYPE="BENCHMARKS"
# export MODEL_NAME="uno_train_improve"
export MODEL_NAME="unorun"
# Used for output:
# Dunedin:
# export CANDLE_DATA_DIR=/usb3/woz/CANDLE_DATA_DIR
# Aurora:
export CANDLE_DATA_DIR=/lus/flare/projects/candle_aesp_CNDA/out
# UNO=$HOME/proj/I-UNO.clean
UNO=$HOME/proj/I-UNO.jw
IMPROVELIB=$HOME/proj/IMPROVE
UNO_TOOLS=$HOME/proj/I-Scratch/IMPROVE_UNO_data
# $THIS is for hed_setup.py
export PYTHONPATH=$UNO:$IMPROVELIB:$THIS:$UNO_TOOLS:${PYTHONPATH:-}

export TURBINE_LEADER_HOOK_STARTUP="$( sed 's/#.*//;s/$/;/' ${THIS}/hook-leader.tcl )"

export UPF_DFLTS=$THIS/hed-dflts.json
export UPF=hed-1.txt

echo CANDLE_MODEL_NAME
set -x
supervisor $SITE upf $CFG_SYS
