#!/bin/bash
set -eu

# POST SH
# Send exported TGZs to Lambda

if ! SUPERVISOR_HOME=$( supervisor -H )
then
  echo "Add supervisor to PATH!"
  exit 1
fi

# For log()/assert():
LOG_NAME="post.sh"
source $SUPERVISOR_HOME/workflows/common/sh/utils.sh

SIGNATURE -H "Provide an output DIR (e.g., .../OUT) and LIST!" \
          DIR LIST - ${*}

EXPS=( $( cat $LIST ) )

cd $( realpath $DIR )

printf "post: %i from: %s\n" ${#EXPS[@]} $( pwd -P )

for EXP in ${EXPS[@]}
do
  rsync EXP$EXP/EXP$EXP.tgz lambda:Public/data/HED
done

echo "OK"
