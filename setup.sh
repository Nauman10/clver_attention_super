#! /bin/bash


cd image && python clevr_split.py
python script_clevr_annotations.py
cd ..
python resize.py




cd diff_attempt && python get_diff.py
python resize2.py
python clevr_split.py


cd ./../ && python prepro.py 

