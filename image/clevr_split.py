import os
from shutil import copyfile

train_size = 16000
validate_size = 4000


if not os.path.exists('./train2014/'):
	os.makedirs('./train2014/')

if not os.path.exists('./val2014/'):
        os.makedirs('./val2014')

for index,item  in enumerate(sorted(os.listdir('./output/images/'))):
	if item.endswith('.png') and index < train_size:
                print index,item
		copyfile('./output/images/'+item,'./train2014/'+item)
	elif item.endswith('.png') and index >=  train_size and index < (validate_size+train_size):
		print index,item
		copyfile('./output/images/'+item,'./val2014/'+item)
	
