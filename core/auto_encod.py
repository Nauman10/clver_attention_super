import tensorflow as tf
import skimage.transform
import numpy as np
import time
import os
import cPickle as pickle
from scipy import ndimage
from utils import *
import pdb
import hickle



training_size=16000
learning_rate =0.01
batch_size = 100
num_steps = 30000 #not sure about this

image_encoding=4096
text_raw = 8

numb_output_pixels = 4096

# Setting up the training data

data_path_train ='./data/train'
data_train={}


train_label_path='./diff_attempt/diff_train_images/'


            
with open(os.path.join(data_path_train, 'train.captions.pkl'), 'rb') as f:
    data_train['captions']=pickle.load(f)
with open(os.path.join(data_path_train, 'train.features.hkl'), 'rb') as f:
    data_train['features'] = hickle.load(f)

list_train_files=os.listdir(train_label_path)

image_files=np.array(map(lambda x: ndimage.imread(train_label_path+x,mode='L'),sorted(list_train_files)))

train_image_file_labels = np.zeros(training_size,numb_output_pixels)

for i in range(0,int(training_size),2):
    train_image_file_labels[i:i+1,:] = image_files[int(i/2)].flatten()

# Setting up the valadation data

data_path_val ='./data/val'
data_val={}


val_label_path='./diff_attempt/diff_val_images/'
validation_size=400

with open(os.path.join(data_path_val, 'val.captions.pkl'), 'rb') as f:
    data_val['captions']=pickle.load(f)
with open(os.path.join(data_path_val, 'val.features.hkl'), 'rb') as f:
    data_val'features'] = hickle.load(f)

list_val_l_files=os.listdir(val_label_path)

image_files = 0
image_files=np.array(map(lambda x: ndimage.imread(val_label_path+x,mode='L'),sorted(list_val_l_files)))

val_image_file_labels = np.zeros(validation_size,numb_output_pixels)

for i in range(0,validation_size,2):
    val_image_file_labels[i:i+1,:] = image_files[int(i/2)].flatten()




pdb.set_trace()

text_encoding=2048

hidden_1 = 2000
hidden_2= 500
hidden_3= 2000
hidden_4= 4096

X_I = tf.placeholder("float",[batch_size,4096])
X_T = tf.placeholder("float",[batch_size,4096])
Y = tf.placeholder("float",[batch_size,numb_output_pixels])


weights = {
        'text_embedding_w': tf.Variable(tf.random_normal([text_raw,text_encoding]))
        'encoder_w1': tf.Variable(tf.random_normal([image_encoding+text_encoding,hidden_1]))
        'encoder_w2': tf.Variable(tf.random_normal([hidden_1,hidden_2]))
        'encoder_w3': tf.Variable(tf.random_normal([hidden_2,hidden_3]))
        'decoder_w1': tf.Variable(tf.random_normal(hidden_3, hidden_2]))
        'decoder_w2': tf.Variable(tf.random_normal(hidden_2, hidden_1]))
        'decoder_w3': tf.Variable(tf.random_normal(hidden_1, image_encoding]))
}

biases = {
        'text_embedding_b': tf.Variable(tf.random_normal([text_encoding]))
        'encoder_b1': tf.Variable(tf.random_normal([hidden_1]))
        'encoder_b2': tf.Variable(tf.random_normal([hidden_2]))
        'encoder_b3': tf.Variable(tf.random_normal([hidden_3]))
        'decoder_b1': tf.Variable(tf.random_normal( hidden_2]))
        'decoder_b2': tf.Variable(tf.random_normalhidden_1]))
        'decoder_b3': tf.Variable(tf.random_normal(hidden_1, image_encoding]))
}



def get_batch(data):
	
        
def text_embedding(x):
    layer_1= tf.nn.sigmoid(tf.add(tf.matmul(x,weights['text_embedding_w']),biases['text_embedding_b'])
    return layer_1

def encoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmu(x,weights['encoder_w1']),biases['encoder_b1'])
    
    layer2 = tf.nn.sigmoid(tf.add(tf.matmu(layer1,weights['encoder_w2']),biases['encoder_b2'])

    layer3 = tf.nn.sigmoid(tf.add(tf.matmu(layer2,weights['encoder_w3']),biases['encoder_b3']) 
    
    return layer3

def decoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmu(x,weights['decoder_w1']),biases['decoder_b1'])
     
    layer2 = tf.nn.sigmoid(tf.add(tf.matmu(layer1,weights['decoder_w2']),biases['decoder_b2'])

    layer3 = tf.nn.sigmoid(tf.add(tf.matmu(layer2,weights['decoder_w3']),biases['decoder_b3']) 

    return layer3

embedding = text_embedding(X_T)
encoder_output = encoder(tf.add(embedding,X_I))
decoder_output = decoder(encoder_output)

y_pred = decoder_output
y_true = Y

loss = tf.reduce_mean(tf.pow(y_pred-y_true,2))

optimizer = tf.train.GradientDescentOptimizer(learning_rate)


with tf.Graph().as_default:
    sess =tf.Session()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(1,training_size):
	    
