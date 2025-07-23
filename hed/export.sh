#!/bin/bash
set -eu

# EXPORT SH
# Create TGZ with key experiment inputs and outputs

if ! SUPERVISOR_HOME=$( supervisor -H )
then
  echo "export.sh: add supervisor to PATH!"
  exit 1
fi

# For log():
LOG_NAME="export.sh"
source $SUPERVISOR_HOME/workflows/common/sh/utils.sh

SIGNATURE -H "Provide an output DIR (e.g., .../X042/out)!" \
          DIR - ${*}

assert-exists -d $DIR

OUT=$( dirname  $DIR )
EXP=$( basename $DIR )
cd $OUT

log "DIR:" $DIR

assert-exists $EXP/scores.csv

renice -n 19 ${$} > /dev/null

QUERY=( -name '*scores.json'   -or
        -name parameters.txt   -or
        -name '*predicted.csv' )

RESULTS=( $( find $EXP ${QUERY[@]} | sort ) )

FILES=( $EXP/jobid.txt
        $EXP/hed-*.json
        $EXP/dflts-*.json
        $EXP/scores.csv
        $EXP/turbine.log
        $EXP/turbine-env.txt
        $EXP/turbine-pbs.sh
        ${RESULTS[@]} )

TGZ=$DIR/$EXP.tgz
log "packing ${#FILES[@]} inputs ..."
tar cfz $TGZ ${FILES[@]}
log "wrote:" $( basename $TGZ ) 
