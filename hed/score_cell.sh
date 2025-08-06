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

if ! top_drugs.sh $DIR true $CELL 10 > $TMPDIR/drug-$CELL-true.list
then
  abort "failed on:" $TMPDIR/drug-$CELL-true.list
fi
if ! top_drugs.sh $DIR pred $CELL 10 > $TMPDIR/drug-$CELL-pred.list
then
  abort "failed on:" $TMPDIR/drug-$CELL-pred.list
fi

# cat drug-$CELL-true.list
# echo :::
# cat drug-$CELL-pred.list

list_match.py $TMPDIR/drug-$CELL-{true,pred}.list > $TMPDIR/drug-$CELL-match.txt
