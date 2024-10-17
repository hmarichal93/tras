#!/bin/bash
# Ensure the script stops if any command fails
#set -e
# Check if the script is being run with a parameter for the root directory
if [ $# -ne 1 ]; then
  echo "Usage: $0 $CONDA_PREFIX_1, where $CONDA_PREFIX_1 is the path to the conda installation directory"
  exit 1
fi
PORT=8501
source $1/etc/profile.d/conda.sh
conda activate dendrotool

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $SCRIPT_DIR
export PYTHONPATH="./automatic_methods/tree_ring_delineation/mlbrief_inbd:$PYTHONPATH"
streamlit run app.py  --server.port $PORT &
#open browser at http://localhost:8501/
open