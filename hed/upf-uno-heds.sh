#!/bin/bash
set -eu

# UPF UNO HEDS

usage()
{
  echo "usage: upf-uno-hed.sh SITE UPF DFLTS [RESTART]"
}

case ${#} in
  3) SITE=$1 UPF=$2 DFLTS=$3 RESTART="" ;;
  4) SITE=$1 UPF=$2 DFLTS=$3 RESTART=$4 ;;
  *) usage
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
THIS=$(            realpath $( dirname $0 ) )
SUPERVISOR_HOME=$( supervisor -H )
WORKFLOWS_ROOT=$(  realpath $SUPERVISOR_HOME/workflows )
source $WORKFLOWS_ROOT/common/sh/utils.sh
sv_path_prepend $WORKFLOWS_ROOT/common/sh

# Project software/data locations:
CANDLE=/lus/flare/projects/candle_aesp_CNDA
SFW=$CANDLE/wozniak/proj

# Main workflow info:
export CANDLE_FRAMEWORK="keras"
export CANDLE_MODEL_TYPE="BENCHMARKS"
export MODEL_NAME="unorun"
export OBJ_RETURN="val_loss"
CFG_SYS=cfg-sys-1.sh

# Data setup:
# Inputs on permanent FS:
# Small data:
export DATA_SOURCE=$SFW/I-UNO.jw/exp_result
# Big data:
# export DATA_SOURCE=$CANDLE/wozniak/raw_data_3-26-2025-preprocessed
# Used for output:
# Dunedin:
# export CANDLE_DATA_DIR=/usb3/woz/CANDLE_DATA_DIR
# Aurora:
export CANDLE_DATA_DIR=$CANDLE/out

# Software locations:
# UNO=$HOME/proj/I-UNO.clean
UNO=$SFW/I-UNO.jw
IMPROVELIB=$SFW/IMPROVE
I_SCRATCH=$( realpath $THIS/.. )
UNO_TOOLS=$I_SCRATCH/IMPROVE_UNO_data
MPL=$SFW/mae_poly_loss
# $THIS is for hed_setup.py
export PYTHONPATH=$UNO:$IMPROVELIB:$THIS:$UNO_TOOLS:$MPL:${PYTHONPATH:-}
export IMPROVE_LOG_LEVEL=WARN

# Swift/Turbine/ADLB settings:
export TURBINE_LEADER_HOOK_STARTUP="$( sed 's/#.*//;s/$/;/' ${THIS}/hook-leader.tcl )"
export TURBINE_PRELAUNCH="source $THIS/prelaunch.sh"
# export TURBINE_LOG=1 TURBINE_DEBUG=1 \
#        ADLB_DEBUG=1 ADLB_DEBUG_RANKS=1 ADLB_DEBUG_HOSTMAP=1
export ADLB_DEBUG_RANKS=1 ADLB_DEBUG_HOSTMAP=1

# Run the workflow!
set -x
supervisor $SITE upf $CFG_SYS
