#! /bin/bash

source /home/naumanahad/tensor_flow/bin/activate
python ./get_diff.py
cd ./../  && python ./resize2.py
cd ./diff_attempt && python  ./clevr_split.py 
