#!/bin/zsh
set -eu

which pip
read -t 3 || true
PKGS=(
  scikit-learn matplotlib pandas
  "tensorflow==2.15"
)

set -x
pip install $PKGS
