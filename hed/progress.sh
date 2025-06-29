#!/bin/bash
set -eu

# PROGRESS SH
# Check progress of running or completed workflow

if ! SUPERVISOR_HOME=$( supervisor -H )
then
  log "Add supervisor to PATH!"
  exit 1
fi

# For log():
LOG_NAME="progress.sh"
source $SUPERVISOR_HOME/workflows/common/sh/utils.sh

SIGNATURE -H "Provide an output DIR (e.g., .../X042/out)!" \
          DIR - ${*}

assert-exists -d $DIR

log "DIR: $DIR"

OUTS=( $DIR/out/out-*.txt )
echo "OUTs: ${#OUTS[@]}"

echo -n "RUNS: "
cat ${OUTS[@]} | grep -c "run_tensorflow.*pkg.*\.\.\."

echo -n "DONE: "
cat ${OUTS[@]} | grep -c "run_tensorflow.*done"

MKRS=( $DIR/markers/marker-*.txt )
echo "MKRS:" ${#MKRS[@]}
