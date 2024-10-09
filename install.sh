#!/bin/bash

# Ensure the script stops if any command fails
set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 $CONDA_PREFIX_1 , where $CONDA_PREFIX_1 is the path to the conda installation directory"
  exit 1
fi

ROOT_DIRECTORY=$PWD

sudo apt install gnome-terminal -y
pip install -r requirements.txt

git submodule set-url automatic_methods/tree_ring_delineation/cstrd_ipol https://github.com/hmarichal93/cstrd_ipol.git
git submodule set-url automatic_methods/tree_ring_delineation/mlbrief_inbd https://github.com/hmarichal93/mlbrief_inbd.git

# Install repository dependencies
echo "Installing repository dependencies..."
if git submodule update --init; then
    echo "Repository dependencies installed successfully."
else
    echo "Error installing repository dependencies."
    exit 1
fi

# Install INBD project dependencies
echo "Installing INBD dependencies..."
cd "./automatic_methods/tree_ring_delineation/mlbrief_inbd" || { echo "Directory not found: ./automatic_methods/tree_ring_delineation/mlbrief_inbd"; exit 1; }
if git submodule update --init; then
    echo "INBD dependencies installed successfully."
else
    echo "Error installing INBD dependencies."
    exit 1
fi

# Install CS-TRD project dependencies
echo "Installing CS-TRD dependencies..."
cd "../cstrd_ipol" || { echo "Directory not found: ../cstrd_ipol"; exit 1; }
if git submodule update --init && git checkout early_late_wood; then
    echo "CS-TRD dependencies installed successfully."
else
    echo "Error installing CS-TRD dependencies."
    exit 1
fi

# Compile the CS-TRD project
echo "Compiling devernay_1.0..."
cd ./externas/devernay_1.0 || { echo "Directory not found: ./externas/devernay_1.0"; exit 1; }
if make clean && make; then
    echo "devernay_1.0 compiled successfully."
else
    echo "Error compiling devernay_1.0."
    exit 1
fi

echo "Installation complete and environment ready."


########################################################################################################################
#Create desktop Icon

# Navigate to the root directory
cd "$ROOT_DIRECTORY" || { echo "Directory not found: $ROOT_DIRECTORY"; exit 1; }

# Define the EXEC_COMMAND based on the current directory
EXEC_COMMAND="$ROOT_DIRECTORY/run.sh $1"
echo $EXEC_COMMAND
# Path to the .desktop file (create on Desktop for testing)
DESKTOP_FILE_PATH="$HOME/Desktop/dendrotool.desktop"

# Define the icon path
ICON_PATH="$ROOT_DIRECTORY/assets/wood_main.png"

#chmod +x "$EXEC_COMMAND"

# Content of the .desktop file
echo "[Desktop Entry]
Version=1.0
Name=Dendrotool
Comment=Launch the Dendrotool Streamlit App
Exec=gnome-terminal -- bash -c '$EXEC_COMMAND; exec bash'
Icon=$ICON_PATH
Terminal=true
Type=Application
Categories=Development;Science;
" > "$DESKTOP_FILE_PATH"

# Make the .desktop file executable
chmod +x "$DESKTOP_FILE_PATH"

# Mark the .desktop file as trusted (to enable "Allow Launching" automatically)
gio set "$DESKTOP_FILE_PATH" metadata::trusted true

# Provide feedback to the user
echo "The .desktop file has been created at: $DESKTOP_FILE_PATH and marked as a trusted launcher."

