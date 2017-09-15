"""Description
Creates LMDB from image files and labels read from csv file below: 
/data4/plankton_wi17/mpl/target_domain/code_dataset_organization/ogspc-fourhr_target_image_path_labels.csv
Target Version
"""
from __future__ import print_function
import os
import random
import sys
import lmdb
import caffe
import numpy as np
import glob
import pandas as pd
import cv2

lab_google_root = '/data4/plankton_wi17/plankton/compare_google_lab/copepod_order/spctarget'
TARGET_ROOT = '/data4/plankton_wi17/mpl/target_domain'
GENERAL_CODE_PATH = '/data4/plankton_wi17/mpl/target_domain/code_dataset_organization'
GENERAL_IMAGES_PATH = '/data4/plankton_wi17/mpl/target_domain/image_sym'
DEBUG = False


def write_caffe_lmdb(img_fns, labels, lmdb_fn):
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
        img = (img*255).astype(np.uint8)        # [0,1]->[0,255]
        img = aspect_resize(img)
        # img = caffe.io.resize_image(img, np.array([256, 256]))     # Resize to 256
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


def aspect_resize(im):
    '''
    Preserves aspect ratio and resizes image
    :param im: data array of image that is rescaled 0->255
    :return: resized image
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

def read_fns_from_csv():

    # Open csv file and convert files and labels to list
    ogspc_csv = '/data4/plankton_wi17/mpl/target_domain/code_dataset_organization/ogspc-fourhr_target_image_path_labels.csv'
    df = pd.read_csv(ogspc_csv)

    # Shuffle dataframe
    df_shuffled = df.iloc[np.random.permutation(len(df))]
    df_shuffled = df_shuffled.reset_index(drop=True)

    # Preprocess csv file for files and labels
    fns_all = df_shuffled['img'].tolist()
    lbs_all = df_shuffled['img_label'].tolist()
    imgid_all = df_shuffled['img_id'].tolist()
    lbl_count = df_shuffled.img_label.value_counts()
    if DEBUG:
        print(lbl_count)

    # Replace string with numeric labels
    for index, lbl in enumerate(lbs_all):
        if lbl.startswith("[u'Copepod']"):
            lbs_all[index] = 0
        else:
            lbs_all[index] = 1
    lbs_all = np.asarray(lbs_all) # convert list to array

    # Add source path to each image
    src_path = '/data4/plankton_wi17/mpl/target_domain/image_sym'
    fns_all = [os.path.join(src_path, img_fn) for img_fn in fns_all]

    # Check if first image can be found to test for entire list
    if not os.path.exists(fns_all[0]):
        print("Does not exist")

    return fns_all, lbs_all, imgid_all

def main():

    # Check if main path to images exists
    if not os.path.exists(GENERAL_IMAGES_PATH):
        print("DirNotFoundError: Main path to images not found")
        return

    # Get file names and labels
    test1_fns, test1_lbs, test1_id = read_fns_from_csv()
    with open(TARGET_ROOT + "/aspect_target_image_path_labels.txt",'w') as f:
        for i in range(len(test1_fns)):
            f.write(str(test1_fns[i] + ";" + str(test1_id[i]) + ";" + str(test1_lbs[i]) + '\n'))
    f.close()

    '''
    Modified code to add more training data to spcinsitu image classifier
    '''
    # with open("target_add2training.txt","w") as f:
    #     nSmpl = test1_lbs.size
    #     for i in range(nSmpl):
    #         f.write(str(test1_fns[i]) + ' ' + str(test1_lbs[i]) + '\n')
    # f.close()
    # if os.path.exists("target_add2training.txt"):
    #     print("yes")

    write_caffe_lmdb(test1_fns, test1_lbs, os.path.join(TARGET_ROOT,'aspect_target_bootstrap.LMDB'))

if __name__ == '__main__':
    main()

