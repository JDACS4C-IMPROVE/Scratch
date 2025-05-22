#!/bin/bash
set -eu

# UPF UNO HEDS

if (( ${#} != 1 ))
then
  echo "usage: upf-uno-heds.sh SITE"
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
CFG_SYS=$THIS/cfg-sys-1.sh

export CANDLE_FRAMEWORK="keras"
export CANDLE_MODEL_TYPE="BENCHMARKS"
# export MODEL_NAME="uno_train_improve"
export MODEL_NAME="uno"
# Used for output:
export CANDLE_DATA_DIR=/usb3/woz/CANDLE_DATA_DIR
UNO=$HOME/proj/IMPROVE-UNO
IMPROVELIB=$HOME/proj/IMPROVE
# $THIS is for hed_setup.py
export PYTHONPATH=$UNO:$IMPROVELIB:$THIS:${PYTHONPATH:-}

export UPF_DFLTS=$THIS/hed-dflts.json
export UPF=$THIS/hed-1.txt

echo CANDLE_MODEL_NAME
supervisor dunedin upf $CFG_SYS
# $EMEWS_PROJECT_ROOT/swift/workflow.sh $SITE -a $CFG_SYS $THIS/hed-1.txt
