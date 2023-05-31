#!/usr/bin/env bash
source ./env/bin/activate
python3 src/log_reg.py
python3 src/nn.py
deactivate