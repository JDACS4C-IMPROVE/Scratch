#!/bin/bash
set -eu

# PROGRESS SH
# Check progress of running or completed workflow

if ! SUPERVISOR_HOME=$( supervisor -H )
then
  echo "Add supervisor to PATH!"
  exit 1
fi

# For log()/assert():
LOG_NAME="progress.sh"
source $SUPERVISOR_HOME/workflows/common/sh/utils.sh

SIGNATURE -H "Provide an output DIR (e.g., .../X042/out)!" \
          DIR - ${*}

assert-exists $DIR $DIR/jobid.txt $DIR/output.txt

echo   "DIR:  $DIR"
printf "JOB:  "
sed 's/\..*//' $DIR/jobid.txt

# Find JSON:
JSON=( $DIR/dflts-*.json )
JSONS=${#JSON[@]}
assert $(( JSONS == 1 )) "Found $JSONS dflts JSONs!"
echo "JSON: ${JSON[@]}"

# Report metadata:
grep "epochs\|alpha" $JSON | sed 's/  //;s/[",]//g'

# set -x
# Report job run stats:
if grep -q "walltime .* exceeded limit" $DIR/output.txt
then
  printf "STOP:"
  grep "walltime .* exceeded limit" $DIR/output.txt | \
    cut -d : -f 3
else
  # Pull out the end of the main log:
  TAIL=$( mktemp --suffix=.txt /tmp/$USER/tail-XXX )
  tail $DIR/output.txt > $TAIL

  if ! grep -q "CODE: *0" $TAIL
  then
    grep "CODE:" $TAIL
    rm $TAIL
    abort "non-zero exit code!"
  fi

  printf "TIME: "
  grep "MPIEXEC TIME:" $TAIL | awk '{ print    $3 }'
  printf "DONE: "
  grep "COMPLETED:"    $TAIL | awk '{ print $2 " " $3 }'
  rm $TAIL
fi

# Report counts...

printf "OUTs: "
OUTS=( $DIR/out/out-*.txt )
echo ${#OUTS[@]}

echo -n "NEWS: "
cat ${OUTS[@]} | grep -c "run_tensorflow.*pkg.*\.\.\."

echo -n "DONE: "
cat ${OUTS[@]} | grep -c "run_tensorflow.*done"

echo -n "RUNS: "
RUNS=( $DIR/run/* )
echo ${#RUNS[@]}

MKRS=( $DIR/markers/marker-*.txt )
echo "MKRS:" ${#MKRS[@]}
