#!/bin/bash
set -eu

# SCORE CELL
# Score the preds for this cell

if ! SUPERVISOR_HOME=$( supervisor -H )
then
  echo "Add supervisor to PATH!"
  exit 1
fi

# For log()/assert():
LOG_NAME="score_cell.sh"
source $SUPERVISOR_HOME/workflows/common/sh/utils.sh

SIGNATURE -H "Provide DIR" \
          DIR CELL - ${*}

assert-exists $DIR $DIR/jobid.txt

TMPDIR=/tmp/$USER
mkdir -pv $TMPDIR

top_drugs.sh $DIR true $CELL 10 > drug-$CELL-true.list
top_drugs.sh $DIR pred $CELL 10 > drug-$CELL-pred.list

cat drug-$CELL-true.list
echo :::
cat drug-$CELL-pred.list

list_match.py drug-$CELL-{true,pred}.list
