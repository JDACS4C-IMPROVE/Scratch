#!/bin/zsh
set -eu

THIS=${0:A:h}
mkdir -pv $THIS/out

A=(
  -l nodes=1:ppn=2
  -l walltime=5:00
  -A candle_aesp_CNDA
  -q debug
  -l filesystems=home:flare
  -o $THIS/out/out-$( date "+%Y-%m-%d_%H:%M" ).txt
  -j oe
)

set -x
qsub $A $THIS/job.sh
