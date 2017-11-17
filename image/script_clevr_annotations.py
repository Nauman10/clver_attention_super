import json
import os
import pdb

data={}
train_size=8000
validation_size=2000
dir_train_image="./train2014/"
dir_val_image="./val2014/"
dir_captions="./output/qa/"


data_train = {}
data_train['images']=[]
data_train['annotations']=[]


data_val = {}
data_val['images']=[]
data_val['annotations']=[]


for file in sorted(os.listdir(dir_train_image)):
	if file.endswith(".png"):
		data_train['images'].append({'file_name':file,'id':file[-12:-4]})
                with open(dir_captions+file[:-6]+'.json') as json_file:
		        caption_file = json.load(json_file)
 
                        if file[-5] == 'a':
                                print "success"
                        	data_train['annotations'].append({'image_id':str(file[-12:-6]+"_a"),'caption':caption_file['description_a'],'id':str(file[-12:-6]+"_a")})
                        elif file[-5] == 'b':
 				print "success"
                        	data_train['annotations'].append({'image_id':str(file[-12:-6]+"_b"),'caption':caption_file['description_b'],'id':str(file[-12:-6]+"_b")})

for file in sorted(os.listdir(dir_val_image)):
        if file.endswith(".png"):
                data_val['images'].append({'file_name':file,'id':file[-12:-4]})
                with open(dir_captions+file[:-6]+'.json') as json_file:
                        caption_file = json.load(json_file)
			if file[-5] == 'a':
                        	data_val['annotations'].append({'image_id':str(file[-12:-6]+"_a"),'caption':caption_file['description_a'],'id':str(file[-12:-6]+"_a")})
			if file[-5] == 'b':
                        	data_val['annotations'].append({'image_id':str(file[-12:-6]+"_b"),'caption':caption_file['description_b'],'id':str(file[-12:-6]+"_b")})
                




with open('./../data/clevr_train_annotations.json','w') as fp1:
	json.dump(data_train,fp1)

with open('./../data/clevr_val_annotations.json','w') as fp2:
        json.dump(data_val,fp2)
