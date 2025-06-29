#!/bin/bash
set -eu

# UPF UNO HEDS

case ${#} in
  3) SITE=$1 UPF=$2 DFLTS=$3 RESTART="" ;;
  4) SITE=$1 UPF=$2 DFLTS=$3 RESTART=$4 ;;
  *) echo "usage: upf-uno-hed.sh SITE UPF DFLTS [RESTART]"
     exit 1 ;;
esac
export UPF UPF_DFLTS=$DFLTS RESTART

SUPERVISOR_TOOL=$( which supervisor || echo NONE )
if [[ $SUPERVISOR_TOOL == NONE ]]
then
  echo "Put supervisor in PATH!"
  exit 1
fi

# Self-configure
THIS=$(               cd $( dirname $0 ) ; /bin/pwd )
SUPERVISOR_HOME=$( supervisor -H )
WORKFLOWS_ROOT=$(  cd $SUPERVISOR_HOME/workflows ; /bin/pwd )

source $WORKFLOWS_ROOT/common/sh/utils.sh
sv_path_prepend $WORKFLOWS_ROOT/common/sh

export OBJ_RETURN="val_loss"
CFG_SYS=cfg-sys-1.sh

export CANDLE_FRAMEWORK="keras"
export CANDLE_MODEL_TYPE="BENCHMARKS"
export MODEL_NAME="unorun"
# Inputs on permanent FS:
# Small data:
export DATA_SOURCE=/home/wozniak/proj/I-UNO.jw/exp_result
# Big data:
# export DATA_SOURCE=/lus/flare/projects/candle_aesp_CNDA/wozniak/raw_data_3-26-2025-preprocessed
# Used for output:
# Dunedin:
# export CANDLE_DATA_DIR=/usb3/woz/CANDLE_DATA_DIR
# Aurora:
export CANDLE_DATA_DIR=/lus/flare/projects/candle_aesp_CNDA/out
# UNO=$HOME/proj/I-UNO.clean
UNO=$HOME/proj/I-UNO.jw
IMPROVELIB=$HOME/proj/IMPROVE
I_SCRATCH=$( realpath $THIS/.. )
UNO_TOOLS=$I_SCRATCH/IMPROVE_UNO_data
MPL=$HOME/proj/mae_poly_loss
# $THIS is for hed_setup.py
export PYTHONPATH=$UNO:$IMPROVELIB:$THIS:$UNO_TOOLS:$MPL:${PYTHONPATH:-}
export IMPROVE_LOG_LEVEL=WARN

export TURBINE_LEADER_HOOK_STARTUP="$( sed 's/#.*//;s/$/;/' ${THIS}/hook-leader.tcl )"

export TURBINE_PRELAUNCH="source $THIS/prelaunch.sh"

# export TURBINE_LOG=1 TURBINE_DEBUG=1 \
#        ADLB_DEBUG=1 ADLB_DEBUG_RANKS=1 ADLB_DEBUG_HOSTMAP=1

export ADLB_DEBUG_RANKS=1 ADLB_DEBUG_HOSTMAP=1

echo CANDLE_MODEL_NAME
set -x
supervisor $SITE upf $CFG_SYS
