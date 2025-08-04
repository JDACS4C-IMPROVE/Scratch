#!/bin/bash
set -eu

# LIST DRUGS
# Find all unique drugs

if ! SUPERVISOR_HOME=$( supervisor -H )
then
  echo "Add supervisor to PATH!"
  exit 1
fi

# For log()/assert():
LOG_NAME="top_drugs.sh"
source $SUPERVISOR_HOME/workflows/common/sh/utils.sh

SIGNATURE -H "Provide DIR TYPE=true|pred CELL" \
          DIR TYPE CELL - ${*}

assert-exists $DIR $DIR/jobid.txt

TMPDIR=/tmp/$USERq
mkdir -pv $TMPDIR
TMP_SCORES=$TMPDIR/scores.csv

case $TYPE in
  true) FIELD=2 ;;
  pred) FIELD=3 ;;
  *)    abort "unknown TYPE=$TYPE"
esac

find $DIR/run -name test_y_data_predicted.csv | \
  xargs -n 8 -- grep $CELL | \
  cut -b 41-46,73- | \
  tr ':,' ' ' | \
  sort -n -k $FIELD
