#!/bin/bash
set -eu

# LIST CELLS
# Find all unique cells from test predictions

if ! SUPERVISOR_HOME=$( supervisor -H )
then
  echo "Add supervisor to PATH!"
  exit 1
fi

# For log()/assert():
LOG_NAME="list_cells.sh"
source $SUPERVISOR_HOME/workflows/common/sh/utils.sh

SIGNATURE -H "Provide DIR" \
          DIR - ${*}

assert-exists $DIR $DIR/jobid.txt

TMPDIR=/tmp/$USER
mkdir -pv $TMPDIR
TMP_SCORES=$TMPDIR/scores.csv

find $DIR/run -name test_y_data_predicted.csv |  \
  xargs -n 8 -- cat | \
  tr ':,' ' ' | \
  cut -d ' ' -f 3 | \
  sort -u | \
  grep -v labels
