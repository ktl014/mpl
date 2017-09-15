"""Description
"""
from __future__ import print_function
import os
import random
import sys
import lmdb
import caffe
import numpy as np
import pandas as pd

spcinsitu_root = '/data4/plankton_wi17/mpl/source_domain/spcinsitu'
classifier = 'insitu_finetune' # Specify which classifier path to download the LMDB files
DEBUG = False

def main():

    # Check if main path to images exists
    if not os.path.exists(spcinsitu_root+'/all_insitu_images'):
        raise ValueError ('Image dir not found')

    # Get file names and labels
    train_fns, train_lbs, val_fns, val_lbs, test1_fns, test1_lbs = read_fns_from_csv()
    print("train {} images".format(len(train_fns)))
    print("val {} images".format(len(val_fns)))
    print("test {} images".format(len(test1_fns)))


    #train_fns, train_lbs = combine_target2train_dataset()
    print("Files received")

    # Write LMDB files
    write_caffe_lmdb(train_fns, train_lbs, os.path.join(spcinsitu_root,classifier,'code','train.LMDB'))
    write_caffe_lmdb(val_fns, val_lbs, os.path.join(spcinsitu_root,classifier,'code','val.LMDB'))
    write_caffe_lmdb(test1_fns, test1_lbs, os.path.join(spcinsitu_root,classifier,'code','test1.LMDB'))


def write_caffe_lmdb(img_fns, labels, lmdb_fn):
    '''
    Writes LMDB file to input into Caffe network
    :param img_fns: List of image paths
    :param labels: List of Labels
    :param lmdb_fn: Name of LMDB File
    :return: N/A
    '''
    if os.path.exists(lmdb_fn) and not DEBUG:
        raise ValueError(lmdb_fn + ' already exists.')

    nSmpl = labels.size
    map_size = nSmpl*3*256*256*8*1.5
    env_img = lmdb.open(lmdb_fn, map_size=map_size)
    print('Generating dataset lmdb: '+lmdb_fn)
    for i in range(nSmpl):
        # Write image datum
        datum = caffe.proto.caffe_pb2.Datum()

        img = caffe.io.load_image(img_fns[i])   # Read image
        # img = caffe.io.resize_image(img, np.array([256, 256]))     # Resize to 256
        img = (img*255).astype(np.uint8)        # [0,1]->[0,255]
        img = aspect_resize(img)                # Preserve aspect ratio
        img = img[:, :, (2, 1, 0)]              # RGB->BGR
        img = np.transpose(img, (2, 0, 1))      # [X,Y,C]->[C,X,Y]

        if DEBUG:
            print(img.max(), img.min(), img.mean(), img.shape)
            exit(0)

        # Prepare Datum
        datum.channels, datum.height, datum.width = img.shape[0], img.shape[1], img.shape[2]
        datum.data = img.tostring()
        datum.label = int(labels[i])

        with env_img.begin(write=True) as txn:
            txn.put('{:08}'.format(i).encode('ascii'), datum.SerializeToString())
        if i % 1000 == 0:
            print('Samples saved:', i, '/', nSmpl)
            sys.stdout.flush()
    return

def read_fns_from_csv():
    '''
    All images and labels come from csv file
    Path: /data4/plankton_wi17/mpl/source_domain/spcinsitu/dataset_tools/insitu_image_path_labels.csv
    :return: 
    '''

    # Open csv file and convert files and labels to list
    df = pd.read_csv(spcinsitu_root + '/dataset_tools/insitu_image_path_labels.csv') # Original image list from spc website
    if not DEBUG:
        print(df.head())
        print(df.shape)

    # Shuffle dataframe
    df = df.iloc[np.random.permutation(len(df))]
    df = df.reset_index(drop=True)
    if not DEBUG:
        print(df)

    # Obtain image paths and labels from csv file
    copepod = df["img"][df['img_label']=="[u'Copepod']"].tolist() # Copepod images
    non_copepod = df["img"][df['img_label'] == '[]'].tolist () # Non-Copepod images

    # Partition into training, validation, and test set
    train_fns, train_lbs, val_fns, val_lbs, test1_fns, test1_lbs = partition(copepod,non_copepod)

    # Return output to get_fns()
    return train_fns, train_lbs, val_fns, val_lbs, test1_fns, test1_lbs

def partition(class_0,class_1):
    f = open("stats_1.txt","w")
    num_img_class0 = len(class_0)
    num_img_class1 = len(class_1)
    if DEBUG:
        print("Class 0 {}".format(num_img_class0))
        print("Class 1 {}".format(num_img_class1))


    # Construct training set
    train0_img = class_0[:int (num_img_class0 * 0.7)]
    train1_img = class_1[:int (num_img_class1 * 0.7)]

    # Set training set
    train_set = train0_img + train1_img
    train_lbs = [0]*len(train0_img) + [1]*len(train1_img)
    if DEBUG:
        train = zip(train_set,train_lbs)
        print(type(train))
        print('test1: {}'.format(train[0:5]))
        # print(train_lbs)

    # Construct validation set
    val0_img = class_0[int (num_img_class0 * 0.7):int (num_img_class0 * 0.8)]
    val1_img = class_1[int (num_img_class1 * 0.7):int (num_img_class1 * 0.8)]

    # Set validation set
    val_set = val0_img + val1_img
    val_lbs = [0]*len(val0_img)+ [1]*len(val1_img)

    # Construct test set
    test0_img = class_0[int (num_img_class0 * 0.8):]
    test1_img = class_1[int (num_img_class1 * 0.8):]

    # Set test set
    test_set = test0_img + test1_img
    test_lbs = [0]*len(test0_img) + [1]*len(test1_img)

    # Shuffle images and labels
    train_set,train_lbs = randomize_writepaths(train_set, train_lbs, 'train')
    val_set,val_lbs = randomize_writepaths(val_set, val_lbs, 'val')
    test_set,test_lbs = randomize_writepaths(test_set, test_lbs, 'test')

    # Write dataset statistics to text file
    f.write('\t' + "Class 0 {}".format(len(train0_img)) + '\n')
    f.write('\t' + "Class 1 {}".format(len(train1_img)) + '\n')
    f.write("Train {}".format(len(train_set)) + '\n')

    f.write('\t' + "Class 0 {}".format(len(val0_img)) + '\n')
    f.write('\t' + "Class 1 {}".format(len(val1_img)) + '\n')
    f.write("Val {}".format(len(val_set)) + '\n')

    f.write('\t' + "Class 0 {}".format(len(test0_img)) + '\n')
    f.write('\t' + "Class 1 {}".format(len(test1_img)) + '\n')
    f.write("Test {}".format(len(test_set)) + '\n')
    f.close()

    return train_set, train_lbs, val_set, val_lbs, test_set, test_lbs

def randomize_writepaths(fns,lbs,key):

    # Sanity check after partitioning
    if len(fns) != len(lbs):
        raise ValueError (key + " Number of files and labels do not match. {} vs {}".format(len(fns),len(lbs)))
    if DEBUG:
        fn_lb = zip(fns,lbs)
        #print('test2: {}'.format(fn_lb[0:5]))
        tst = 'SPC2-1470202558-809214-001-2076-2504-104-176.jpg'
        for i in fn_lb:
            if i[0] == tst:
                print('test2: {}'.format(i))
    index = range(len(fns))
    random.shuffle(index)
    src_path = spcinsitu_root + '/all_insitu_images'
    fns = [os.path.join(src_path,str(fns[i])) for i in index]
    lbs = np.array([lbs[i] for i in index])
    if DEBUG:
        fn_lb1 = zip(fns,lbs)
        # print('test3: {}'.format(fn_lb1[0:5]))
        tst = 'SPC2-1470202558-809214-001-2076-2504-104-176.jpg'
        for i in fn_lb1:
            if i[0].split('/')[7] == tst:
                print('test2: {}'.format(i))
        exit(0)
        print(lbs.reshape(len(lbs),1))

    # Write img path and gtruth txt files
    path_txt = open (spcinsitu_root + '/' + classifier + '/code/' + key + '.txt', 'w')
    nSmpl = lbs.size
    for i in range (nSmpl):
        path_txt.write (str(fns[i]) + " " + str (lbs[i]) + '\n')
    path_txt.close ()
    return fns,lbs

def combine_target2train_dataset():
    txt_filename = '/data4/plankton_wi17/mpl/target_domain/code_dataset_organization/target_add2training.txt'
    with open(txt_filename,"r") as target_f:
        target_list = target_f.readlines()
    target_f.close()
    # target_img = []
    # target_lbs = []
    target_list = [i.split() for i in target_list]

    # for i in target_list:
    #     target_img[i] = i.split()[0]
    #     target_lbs[i] = i.split()[1]

    txt_filename = '/data4/plankton_wi17/mpl/source_domain/spcinsitu/dataset_tools/train.txt'
    with open(txt_filename,"r") as train_f:
        train_list = train_f.readlines()
    train_f.close()
    train_fns = []
    train_lbs = []
    train_list = [i.split() for i in train_list]

    train_list = train_list + target_list

    for i in train_list:
        train_fns.append(i[0])
        train_lbs.append(i[1])
    if DEBUG:
        if len(train_fns) == len(train_lbs):
            print("match")

        # for i in train_list:
    #     train_img[i] = i.split()[0]
    #     train_lbs[i] = i.split()[1]

    index = range(len(train_fns))
    random.shuffle(index)
    train_fns = [train_fns[i] for i in index]
    train_lbs = np.array([train_lbs[i] for i in index])

    return train_fns, train_lbs

def aspect_resize(im):
    '''
    Preserve aspect ratio and resize the image
    :param im: image read in as array. Should be rescaled as 0-255
    :return: res - resized image
    '''
    ii = 256
    mm = [int (np.median (im[0, :, :])), int (np.median (im[1, :, :])), int (np.median (im[2, :, :]))]
    cen = np.floor (np.array ((ii, ii)) / 2.0).astype ('int')  # Center of the image
    dim = im.shape[0:2]
    if DEBUG:
        print
        "median {}".format (mm)
        print
        "ROC {}".format (cen)
        print
        "img dim {}".format (dim)
        # exit(0)

    if dim[0] != dim[1]:
        # get the largest dimension
        large_dim = max (dim)

        # ratio between the large dimension and required dimension
        rat = float (ii) / large_dim

        # get the smaller dimension that maintains the aspect ratio
        small_dim = int (min (dim) * rat)

        # get the indicies of the large and small dimensions
        large_ind = dim.index (max (dim))
        small_ind = dim.index (min (dim))
        dim = list (dim)

        # the dimension assigment may seem weird cause of how python indexes images
        dim[small_ind] = ii
        dim[large_ind] = small_dim
        dim = tuple (dim)
        if DEBUG:
            print
            'before resize {}'.format (im.shape)
        im = cv2.resize (im, dim)
        half = np.floor (np.array (im.shape[0:2]) / 2.0).astype ('int')
        if DEBUG:
            print
            'after resize {}'.format (im.shape)

        # make an empty array, and place the new image in the middle
        res = np.zeros ((ii, ii, 3), dtype='uint8')
        res[:, :, 0] = mm[0]
        res[:, :, 1] = mm[1]
        res[:, :, 2] = mm[2]

        if large_ind == 1:
            test = res[cen[0] - half[0]:cen[0] + half[0], cen[1] - half[1]:cen[1] + half[1] + 1]
            if test.shape != im.shape:
                res[cen[0] - half[0]:cen[0] + half[0] + 1, cen[1] - half[1]:cen[1] + half[1] + 1] = im
            else:
                res[cen[0] - half[0]:cen[0] + half[0], cen[1] - half[1]:cen[1] + half[1] + 1] = im
        else:
            test = res[cen[0] - half[0]:cen[0] + half[0] + 1, cen[1] - half[1]:cen[1] + half[1]]
            if test.shape != im.shape:
                res[cen[0] - half[0]:cen[0] + half[0] + 1, cen[1] - half[1]:cen[1] + half[1] + 1] = im
            else:
                res[cen[0] - half[0]:cen[0] + half[0] + 1, cen[1] - half[1]:cen[1] + half[1]] = im
    else:
        res = cv2.resize (im, (ii, ii))
        half = np.floor (np.array (im.shape[0:2]) / 2.0).astype ('int')

    if DEBUG:
        print
        'aspect_resize: {}'.format (res.shape)
    return res

if __name__ == '__main__':
    main()
    #combine_target2train_dataset()
    '''
    1. GENERAL_PATH
    2. which files to make lmdb for
    3. path hierarchy
    '''
