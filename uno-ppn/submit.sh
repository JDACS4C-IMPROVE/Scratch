#!/bin/zsh
set -eu

# SUBMIT SH

THIS=${0:A:h}
mkdir -pv $THIS/out

OUT=out/out_$( date "+%Y-%m-%d_%H:%M" ).txt
echo "OUT=$OUT"

A=(
  -l nodes=1:ppn=12
  -l walltime=5:00
  -A candle_aesp_CNDA
  -q debug-scaling
  -l filesystems=home:flare
  -o $THIS/$OUT
  -j oe
)

set -x
qsub $A $THIS/job.sh
