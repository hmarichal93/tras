#!/bin/bash
source $HOME/.basrhc
conda activate dendrotool
export PYTHONPATH="./automatic_methods/tree_ring_delineation/mlbrief_inbd:$PYTHONPATH"
streamlit run app.py