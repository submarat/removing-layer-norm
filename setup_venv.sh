#!/bin/bash

# Name of the virtual environment
VENV_NAME="venv"

# Check if python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv $VENV_NAME

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete! To activate the virtual environment, run: source $VENV_NAME/bin/activate"