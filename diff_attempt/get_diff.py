import scipy.misc
import matplotlib
import pdb
import os

if not os.path.exists('diff_images'):
   os.makedirs('diff_images')


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def to_bin(img,threshold):
    img[img<threshold]=0
    img[img>=threshold]=1
    return 255*img

def get_diff(a,b,name):
    diff=100*abs(a-b) 
    gray_diff = rgb2gray(diff);
    #img= to_bin(gray_diff,70)
    
    #scipy.misc.imsave('./diff_images/%s'%name,img)
      
    scipy.misc.imsave('./diff_images/%s'%name,gray_diff)
     
    print "%s saved"%name

directory='./../image/output/images/'



for file in os.listdir(directory):
    if file.endswith('a.png'):
        
	a = scipy.misc.imread(directory+file)
        
	b = scipy.misc.imread(directory+file[0:-5]+'b.png')
        name = file[0:-5]+'_diff.png'
        get_diff(a,b,name)	





