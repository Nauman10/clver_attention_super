import os
import shutil 

train_size = 8000
validate_size = 2000


if not os.path.exists('diff_train_images'):
        print "here"
	os.makedirs('diff_train_images')
else:
	os.system('rm -rf diff_train_images')


if not os.path.exists('diff_val_images'):
        os.makedirs('diff_val_images')
else:
	os.system('rm -rf diff_val_images')

for index,item  in enumerate(sorted(os.listdir('resized_diff_images'))):
	if item.endswith('.png') and index < train_size:
                print index,item
		shutil.copyfile(os.path.join('resized_diff_images',item),os.path.join('diff_train_images',item))
	elif item.endswith('.png') and index >=  train_size  and index < (validate_size+train_size):
                print index,item
		shutil.copyfile(os.path.join('resized_diff_images',item),os.path.join('diff_val_images',item))
	
