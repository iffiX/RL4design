#!/bin/bash
CONDA_PATH=~/miniconda3

source $CONDA_PATH/bin/activate
conda create -y -n rl4design
conda activate rl4design
# This specific combination may take a while
# minimum cuda version supporting voxcraft is 11.7
# conda install -c nvidia/label/cuda-11.7.0 -c conda-forge cuda=11.7.0 python=3.9.0 gcc=9.4.0 gxx=9.4.0 cmake boost
conda install -y -c nvidia/label/cuda-12.1.1 -c conda-forge cuda=12.1.1 python=3.10.0 gcc=11.3.0 gxx=11.3.0 cmake boost

python -m venv venv
venv/bin/pip install torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
venv/bin/pip install -r requirements.txt
