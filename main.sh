#!/bin/bash

python3 cyclic_regression.py --k=5 --ts=8|tee cyclic_5_8_Z3.txt
python3 cyclic_regression.py --k=5 --ts=16|tee cyclic_5_16_Z3.txt

python3 imagesum.py|tee imagesum.txt

python3 regression_SGD.py --k=5 --gt_subgroup=Zk|tee regression_SGD_Z5.txt
python3 regression_SGD.py --k=5 --gt_subgroup=D2k|tee regression_SGD_D5.txt

python3 linear_regression.py --k=5|tee linear_regression_5_10.txt

