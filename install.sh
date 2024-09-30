#!/bin/bash
echo "Installing dependencies..."
git submodule update --init

echo "Installing INBD dependencies..."
cd "./automatic_methods/tree_ring_delineation/mlbrief_inbd" && git submodule update --init