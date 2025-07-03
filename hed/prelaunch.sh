
# PRELAUNCH SH
# This is sourced by Swift/T in the job script before the
# main workflow.swift starts
# It is specified in the main UI script, upf-uno-heds.sh

# /usr/bin/time --format="UNTAR: %E" ${TURBINE_LAUNCHER} \
#        tar -xm -f /lus/flare/projects/candle_aesp_CNDA/sfw/TF.tar -C /tmp

set -eu

echo "PRELAUNCH.SH ..."

# Turbine installation w/o Python for use of mpi-cp:
TURBINE_PLAIN_HOME=/lus/flare/projects/candle_aesp_CNDA/sfw/aurora/swift-t/2025-06-03-tcl
MPI_CP=$TURBINE_PLAIN_HOME/turbine/bin/mpi-cp

TAR=/lus/flare/projects/candle_aesp_CNDA/sfw/TF.tar
P=( -n $NODES -ppn 1 )
echo "MPI-CP $TAR ..."
/usr/bin/time --format="MPI-CP: %E" \
  ${TURBINE_LAUNCHER} ${P[@]} $MPI_CP $TAR /tmp
echo "UNTAR ..."
/usr/bin/time --format="UNTAR: %E" \
  ${TURBINE_LAUNCHER} ${P[@]} tar -xm -f /tmp/TF.tar -C /tmp

set +eu
module load frameworks
# source /opt/aurora/24.347.0/oneapi/intel-conda-miniforge/etc/profile.d/conda.sh
set -eu
PATH=/tmp/TF/bin:$PATH

# Force rank prefixes in main file output.txt:
export TURBINE_LAUNCH_OPTIONS="-l"

echo "PRELAUNCH.SH DONE."
