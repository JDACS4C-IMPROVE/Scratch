#!/bin/bash
set -eu

# SCORE ALL CELLS
# Score the preds for each cell in all_cells.txt

if ! SUPERVISOR_HOME=$( supervisor -H )
then
  echo "Add supervisor to PATH!"
  exit 1
fi

# For log()/assert():
LOG_NAME="score_all_cells.sh"
source $SUPERVISOR_HOME/workflows/common/sh/utils.sh

SIGNATURE -H "Provide DIR" \
          DIR - ${*}

assert-exists $DIR $DIR/jobid.txt

TMPDIR=/tmp/$USER
mkdir -pv $TMPDIR

set -x
cat all_cells.txt | while true
do
  read CELL || break
  score_cell.sh $DIR $CELL
done
