#!/bin/bash
set -eu
shopt -s nullglob  # Ignore empty globs

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

grep "PROCS\|PPN" $DIR/turbine.log

# Report metadata:
grep "epochs\|alpha" $JSON | sed 's/  //;s/[",]//g'

# Report counts...

assert-exists -d $DIR/out 

printf "OUTs: "
OUTS=( $DIR/out/out-*.txt )
echo ${#OUTS[@]}

echo -n "RUNs: "
RUNS=( $DIR/run/* )
echo ${#RUNS[@]}

if (( ${#OUTS[@]} > 0 ))
then
  echo -n "NEWs: "
  cat ${OUTS[@]} | grep -c "run_tensorflow.*pkg.*\.\.\." || true

  echo -n "DONE: "
  cat ${OUTS[@]} | grep -c "run_tensorflow.*done" || true
fi

MKRS=( $DIR/markers/marker-*.txt )
echo "MKRs:" ${#MKRS[@]}

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

  if ! grep -q "CODE:" $TAIL
  then
    rm $TAIL
    abort "No exit code!"
  fi

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
