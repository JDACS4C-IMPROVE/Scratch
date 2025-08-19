#!/bin/zsh
set -eu

# INSTALL PQ SH
# Installs conda packages needed for the scripts in this directory

PKGS=( fastparquet
       pandas
       scikit-learn
     )

conda install --yes -c conda-forge $PKGS
