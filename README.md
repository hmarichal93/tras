# DendroTool: An Interactive Software for tracing Tree Ring Cross-Sections.
DendroTool is a Python-based interactive software designed for tracing and analyzing tree ring cross-sections. It provides tools for automatic and manual ring delineation, as well as metrics computation. The application leverages Streamlit for the user interface, making it easy to visualize and interact with the data.
Features

**Automatic Ring Delineation**: Automatically detect and trace tree rings.

**Manual Ring Delineation**: Manually adjust and refine ring boundaries.

**Metrics Computation**: Calculate various metrics based on the traced rings.

**Interactive Visualization**: Use Streamlit to visualize and interact with the data.

![Preview Image](assets/preview.png)
## Installation
Follow the instructions below to set up the environment and install the necessary dependencies:
```bash 
conda env create -f environment.yml
conda activate dendrotool
./install.sh $CONDA_PREFIX_1
```
Replace `$CONDA_PREFIX_1` with the path to the conda environment where the application is installed. 
The `install.sh` script will download the necessary data files and set up the application for use. It will create an icon on the desktop for easy access to the application called `DendroTool`.
## Usage
Run the application using the following command:
```bash
./run.sh $CONDA_PREFIX_1
```
Replace `$CONDA_PREFIX_1` with the path to the conda environment where the application is installed. 



