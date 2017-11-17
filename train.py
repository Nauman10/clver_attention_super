import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import skimage.transform
import numpy as np
import time
import os
import scipy.misc
import cPickle as pickle
from scipy import ndimage
#from utils import *
import pdb
import hickle
import matplotlib.pyplot as plt
import json

training_size=1000
learning_rate =0.001
batch_size = 100
epochs = 30000 

image_encoding=4096
text_raw = 12

numb_output_pixels = 16*16

# Setting up the training data


data_train={}

data_path_train=os.path.join('data','train')


train_label_path = os.path.join('diff_attempt','diff_train_images')


target_directory='generated_diff'

if os.path.exists(target_directory):
    os.system("rm -rf {0}/* ".format(target_directory))
 
if not os.path.exists(target_directory):
    os.mkdir(target_directory)

            
with open(os.path.join(data_path_train, 'train.captions.pkl'), 'rb') as f:
    data_train['captions']=pickle.load(f)

data_train['features'] = hickle.load(os.path.join(data_path_train, 'train.features.hkl'))

list_train_files=os.listdir(train_label_path)

image_files=np.array(map(lambda x: ndimage.imread(os.path.join(train_label_path,x),mode='L'),sorted(list_train_files)))

train_image_file_labels = np.zeros((training_size,numb_output_pixels))

for i in range(0,int(training_size),2):
    train_image_file_labels[i,:] = image_files[int(i/2)].flatten()
    train_image_file_labels[i+1,:] = image_files[int(i/2)].flatten()
# Setting up the valadation data


data_val={}
data_path_val = os.path.join('data','val')


val_label_path = os.path.join('diff_attempt','diff_val_images')
#train_image_file_labels[train_image_file_labels < 70] =0
#train_image_file_labels[train_image_file_labels >= 70] =1

train_image_file_labels = train_image_file_labels/255
validation_size=1000

with open(os.path.join(data_path_val, 'val.captions.pkl'), 'rb') as f:
    data_val['captions']=pickle.load(f)

data_val['features'] = hickle.load(os.path.join(data_path_val, 'val.features.hkl'))

list_val_l_files=os.listdir(val_label_path)

image_files = 0
image_files=np.array(map(lambda x: ndimage.imread(os.path.join(val_label_path,x),mode='L'),sorted(list_val_l_files)))

val_image_file_labels = np.zeros((validation_size,numb_output_pixels))

for i in range(0,validation_size,2):
    val_image_file_labels[i,:] = image_files[int(i/2)].flatten()
    val_image_file_labels[i+1,:] = image_files[int(i/2)].flatten()



#val_image_file_labels[val_image_file_labels<70]=0
#val_image_file_labels[val_image_file_labels>=70]=1

val_image_file_labels = val_image_file_labels/255
text_encoding=2048

hidden_1 = 4096
hidden_2=  2048
hidden_3= 1024
hidden_4 = 512
hidden_5 = 256
hidden_6 = 128
hidden_7= numb_output_pixels
#hidden_4= 4096


X_I = tf.placeholder(tf.float32,[batch_size,4096])
X_T = tf.placeholder(tf.float32,[batch_size,text_raw])
Y = tf.placeholder(tf.float32,[batch_size,numb_output_pixels])


weights = {

        'text_embedding_w': tf.Variable(tf.random_normal([text_raw,text_encoding])),
        'encoder_w1': tf.Variable(tf.random_normal([image_encoding+text_encoding,hidden_1])),
        'encoder_w2': tf.Variable(tf.random_normal([hidden_1,hidden_2])),
        'encoder_w3': tf.Variable(tf.random_normal([hidden_2,hidden_3])),
        'decoder_w1': tf.Variable(tf.random_normal([hidden_3, hidden_4])),
        'decoder_w2': tf.Variable(tf.random_normal([hidden_4, hidden_5])),
        'decoder_w3': tf.Variable(tf.random_normal([hidden_5, hidden_6])),
        'decoder_w4': tf.Variable(tf.random_normal([hidden_6, hidden_7])),
      
}

biases = {
        'text_embedding_b': tf.Variable(tf.random_normal([text_encoding])),
        'encoder_b1': tf.Variable(tf.random_normal([hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([hidden_2])),
        'encoder_b3': tf.Variable(tf.random_normal([hidden_3])),
        'decoder_b1': tf.Variable(tf.random_normal([hidden_4])),
        'decoder_b2': tf.Variable(tf.random_normal([hidden_5])),
        'decoder_b3': tf.Variable(tf.random_normal([hidden_6])),
        'decoder_b4': tf.Variable(tf.random_normal([hidden_7])),
}




	
        
def text_embedding(x):
    layer_1= tf.nn.sigmoid(tf.add(tf.matmul(x,weights['text_embedding_w']),biases['text_embedding_b']))
    return layer_1

def encoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encoder_w1']),biases['encoder_b1']))
    
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['encoder_w2']),biases['encoder_b2']))

    layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2,weights['encoder_w3']),biases['encoder_b3'])) 
    
    return layer3

def decoder(x):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decoder_w1']),biases['decoder_b1']))
     
    layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,weights['decoder_w2']),biases['decoder_b2']))

    layer3 = tf.nn.sigmoid(tf.add(tf.matmul(layer2,weights['decoder_w3']),biases['decoder_b3'])) 

    layer4 = tf.nn.sigmoid(tf.add(tf.matmul(layer3,weights['decoder_w4']),biases['decoder_b4']))
    return layer4

embedding = text_embedding(X_T)
encoder_output = encoder(tf.concat([embedding,X_I],axis=1))
decoder_output = decoder(encoder_output)

y_pred = decoder_output
y_true = Y



loss = tf.reduce_mean(tf.pow(y_pred-y_true,2))
#loss= tf.nn.softmax_cross_entropy_with_logits(labels=y_pred, logits= y_pred)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


loss_training=[]
loss_validation=[]

sess =tf.Session()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    for e in range(1, epochs+1):
	for i in range(0,training_size,batch_size):
	    x_i=data_train['features'][i:i+batch_size,:]
	    x_t=data_train['captions'][i:i+batch_size,:text_raw]
	    y = train_image_file_labels[i:i+batch_size,:]
           
            

             
	    g_train, _, l_t = sess.run([y_pred,optimizer,loss],feed_dict={X_I : x_i, X_T : x_t,Y : y})
        
	print "Loss: %f  at epoch: %i"%(np.mean(l_t),e)
	
        if ( e%5000  == 0 ):
	    xv_i=data_val['features'][0:100,:]
	    xv_t=data_val['captions'][0:100,:text_raw]
	    yv = val_image_file_labels[0:100,:]
	   
            sub_dir = 'Epoch_'+str(e) 
            target_path_save = os.path.join(target_directory,sub_dir)
            sub_dir='Epoch_'+str(e)
            if not os.path.exists(target_path_save):
		os.mkdir(target_path_save)       
            
	    g,l_v = sess.run([decoder_output,loss],feed_dict={X_I : xv_i, X_T : xv_t,Y : yv})
	    print "Validation loss: %f  at epoch: %i               Loss again is : %f"%(np.mean(l_v),e,np.mean(l_v))
            loss_training.append(str(l_t))
            loss_validation.append(str(l_v))
            list_images=range(0,100)

            scipy.misc.imsave(os.path.join(target_path_save,'Epoch_'+str(e)+'_'+'training'+'_original_.png'),y[10,:].reshape(16,16)*255)
            
            scipy.misc.imsave(os.path.join(target_path_save,'Epoch_'+str(e)+'_'+'training'+'.png'),g_train[10,:].reshape(16,16)*255)
            map(lambda x: scipy.misc.imsave(os.path.join(target_path_save,'Epoch_'+str(e)+'_'+str(x)+'_original_.png'),yv[x,:].reshape([16,16])*255),list_images)
            map(lambda x: scipy.misc.imsave(os.path.join(target_path_save,'Epoch_'+str(e)+'_'+str(x)+'_.png'),g[x,:].reshape([16,16])*255),list_images)	   



dict_result={'training':loss_training,'test': loss_validation}

with open('results.json','w+') as f:
    json.dump(dict_result,f)	
