#!/bin/bash
conda env create -f environment.yml
conda activate dendrotool

echo "Installing dependencies..."
git submodule update --init

echo "Installing INBD dependencies..."
cd "./automatic_methods/tree_ring_delineation/mlbrief_inbd" && git submodule update --init

echo "Installing CS-TRD dependencies..."
cd "../cstrd_ipol" && git submodule update --init && \
          git checkout early_late_wood && \
          pip install --no-cache-dir -r requirements.txt && \
          cd ./externas/devernay_1.0 && make clean && make


