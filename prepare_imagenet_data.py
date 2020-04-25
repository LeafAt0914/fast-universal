import numpy as np
import os
from scipy.misc import imread, imresize

def validate_arguments(mode):
	mode_contents = ['origi', 'torch', 'caffe']

	if not(mode in mode_contents):
		print ('invalid mode')
		exit (-1)

def process_image(image_path, mode):
    validate_arguments(mode)
    img_size = [256, 256]
    crop_size = [224, 224]

    img = imread(image_path, mode='RGB')

    img = imresize(img,img_size)
    #crop
    img = img[(img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2, (img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2, :]
    
    img = img.astype('float32')

    if mode == 'caffe':
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        #RGB to BGR
        img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
    
    elif mode == 'origi':
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939

    elif mode == 'torch':
        img /= 255.0
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img[..., 0] -= mean[0]
        img[..., 1] -= mean[1]
        img[..., 2] -= mean[2]
        img[..., 0] /= std[0]
        img[..., 1] /= std[1]
        img[..., 2] /= std[2]

    return img

def undo_image(img, mode):
    validate_arguments(mode)
    img_copy = np.copy(img)
    if mode =='caffe':
        img_copy[:,:,[0,1,2]] = img_copy[:,:,[2,1,0]]
        img_copy[:, :, 0] = img_copy[:, :, 0] + 123.68
        img_copy[:, :, 1] = img_copy[:, :, 1] + 116.779
        img_copy[:, :, 2] = img_copy[:, :, 2] + 103.939
    elif mode == 'origi':
        img_copy[:, :, 0] = img_copy[:, :, 0] + 123.68
        img_copy[:, :, 1] = img_copy[:, :, 1] + 116.779
        img_copy[:, :, 2] = img_copy[:, :, 2] + 103.939
    elif mode == 'torch':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_copy[..., 0] = img_copy[..., 0] * std[0]
        img_copy[..., 1] = img_copy[..., 1] * std[1]
        img_copy[..., 2] = img_copy[..., 2] * std[2]
        img_copy[..., 0] = img_copy[..., 0] + mean[0]
        img_copy[..., 1] = img_copy[..., 1] + mean[1]
        img_copy[..., 2] = img_copy[..., 2] + mean[2]
        img_copy = img_copy * 255.0

    return img_copy

#different CNN with different mode will get universal perturbations with different format,
#we use undo_UAP to process universal perturbations to an origin format 
#and use process_UAP to process origin format universal perturbations to target format.
def process_UAP(UAP, mode):
    validate_arguments(mode)
    UAP_copy = np.copy(UAP)
    if mode == 'origi':
        pass
    elif mode == 'caffe':
        UAP_copy[:,:,[0,1,2]] = UAP_copy[:,:,[2,1,0]]
    elif mode == 'torch':
        std = [0.229, 0.224, 0.225]        
        UAP_copy[..., 0] /= (0.229*255)
        UAP_copy[..., 1] /= (0.224*255)
        UAP_copy[..., 2] /= (0.225*255)

    return UAP_copy

def undo_UAP(UAP, mode):
    validate_arguments(mode)
    UAP_copy = np.copy(UAP)
    if mode == 'origi':
        pass
    elif mode == 'caffe':
        UAP_copy[:,:,[0,1,2]] = UAP_copy[:,:,[2,1,0]]
    elif mode == 'torch':
        std = [0.229, 0.224, 0.225]        
        UAP_copy[..., 0] *= (0.229*255)
        UAP_copy[..., 1] *= (0.224*255)
        UAP_copy[..., 2] *= (0.225*255)

    return UAP_copy

def create_imagenet_npy(path_train_imagenet, mode, len_batch=10000):
    validate_arguments(mode)
    sz_img = [224, 224]

    num_channels = 3
    num_classes = 1000

    im_array = np.zeros([len_batch] + sz_img + [num_channels], dtype=np.float32)
    num_imgs_per_batch = int(len_batch / num_classes)

    dirs = [x[0] for x in os.walk(path_train_imagenet)]
    dirs = dirs[1:]

    # Sort the directory in alphabetical order (same as synset_words.txt)
    dirs = sorted(dirs)

    it = 0
    Matrix = [0 for x in range(1000)]

    for d in dirs:
        for _, _, filename in os.walk(d):
            Matrix[it] = filename
        it = it+1

    it = 0
    # Load images, pre-process, and save
    for k in range(num_classes):
        for u in range(num_imgs_per_batch):
            if it % 10 == 0:
                print('Processing image number ', it)
            path_img = os.path.join(dirs[k], Matrix[k][u])
            x = process_image(image_path=path_img, mode=mode)
            x = np.expand_dims(x, axis=0)
            im_array[it:(it+1), :, :, :] = x
            it = it + 1

    return im_array
