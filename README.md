#*Text conditioned image attention**

The repository trains a  model for generating an image  attention map conditioned on a caption.  
The model is trained on the CLVER dataset.

##Requirements

Tensorflow 1.4
Matlplotlib
sklearn
scipy

*Tensorflow 1.4 for GPU  requires Cuda8 and CuDNN 6.0*


Other requirements can be seen in the requirements.txt file




##Setup

Clone this and  pycocoevalcap repos in the same directory

''' 
git clone https://github.com/Nauman10/clver_attention_super.git
git clone https://github.com/tylin/coco-caption.git
'''

To download the required files (dataset and the trained-vgg network) aswell as setting up the 
dictionaries, labels, training splits

'''
./download.sh
./setup.bash
'''



*This repo  is based on the tensor flow implementation at ([https://github.com/yunjey/show-attend-and-tell.git])* 




