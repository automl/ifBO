#!/usr/bash

# Get the absolute path of the script
PATH=$(cd "$(dirname "$0")" && pwd -P)

SRC_PATH=$PATH"/src"

echo "The script is located at: "$SRC_PATH

# Install the required packages/sub-directories
for dir in $(ls -d $SRC_PATH/*/); do
	echo "Installing packages in: "$dir
	pip install -e $dir
done

# Install the core Python packages
pip install -r core_requirements.txt
