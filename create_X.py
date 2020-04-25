import tarfile
import os
import sys
import random
import numpy as np
from prepare_imagenet_data import create_imagenet_npy

'''
choose 10000 images from ILSVRC2012 training set with 10 images per class
'''

#Pos_zip = "E:\\ILSVRC2012\\ILSVRC2012_img_train"
Pos_unzip = os.path.join('datasets', 'ILSVRC2012_train')

def validate_arguments(mode):
	mode_contents = ['origi', 'torch', 'caffe']

	if not(mode in mode_contents):
		print ('invalid mode')
		exit (-1)

def getTenRandomInts(Range):
    res = []
    i = 0
    while(i < 10): #10 images per class
        r = random.randint(0, Range - 1)
        if res.count(r) == 0:
            res.append(r)
            i+=1
    return res

def un_zip(Pos_zip):
    tarfiles = os.listdir(Pos_zip)
    for f in tarfiles:
        filePath = os.path.join(Pos_zip, f)
        if os.path.isfile(filePath) :
            if os.path.splitext(filePath)[1]==".tar":
                tarFile = tarfile.open(filePath,'r')
                print(filePath)
                tarnames = tarFile.getnames()
                print(len(tarnames))
                choose = getTenRandomInts(len(tarnames))
                print(choose)
                for i in choose:
                    tarFile.extract(tarnames[i], Pos_unzip + "\\" + os.path.splitext(f)[0])
    print("unzip is accomplished！")

def Create_X(pathTo_ILSVRC2012,mode):

    validate_arguments(mode)
    un_zip(os.path.join(pathTo_ILSVRC2012, 'ILSVRC2012_img_train'))
    datafile = os.path.join('data', 'imagenet_data_' + str(mode) + '_mode.npy')
    print('>> Creating pre-processed imagenet data...')
    X = create_imagenet_npy(Pos_unzip, mode = mode)
    print('>> Saving the pre-processed imagenet data')
    # Save the pre-processed images
    # Caution: This can take take a lot of space. Comment this part to discard saving.
    np.save(datafile, X)
    return X
