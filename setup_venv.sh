#!/usr/bin/env bash

python3 -m venv ../venv
source ../venv/bin/activate

pip install --upgrade pip

# uncomment to pick a particular torch/cuda version
# pip install -U torch==1.8.1+cu111 torchvision==0.8.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# either get two4two repo
# git clone https://github.com/mschuessler/two4two/
# or link to your local copy
# ln -s your/two4two/repo ./two4two
pip install -e two4two
pip install -e .
