#! /bin/bash

wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -P data/

echo "Downloading clver zip file"
python core/fetch_zip.py 0B1l2A1ZlLu4bS3IyRmNzTzJNWkU clver_set.zip
echo "Download completed"


unzip clver_set.zip -d image


