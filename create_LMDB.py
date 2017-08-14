"""Description
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

lab_google_root = '/data4/plankton_wi17/plankton/compare_google_lab/copepod_order/lab'
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
        img = caffe.io.resize_image(img, np.array([256, 256]))     # Resize to 256
        img = (img*255).astype(np.uint8)        # [0,1]->[0,255]
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

def check_caffe_lmdb(lmdb_fn):
    env = lmdb.open(lmdb_fn, readonly=True)
    print("Opening " + str(lmdb_fn))
    with env.begin() as txn:
        raw_datum = txn.get(b'00000000')
    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)

    flat_x = np.fromstring(datum.data, dtype=np.uint8)
    x = flat_x.reshape(datum.channels,datum.height,datum.width)
    y = datum.label
    print("X label: " + str(x))
    print("Y label: " + str(y))

def read_fns_from_csv():

    # Open csv file and convert files and labels to list
    df = pd.read_csv('fourhr_target_image_path_labels.csv')

    # Shuffle dataframe
    df_shuffled = df.iloc[np.random.permutation(len(df))]
    df_shuffled = df_shuffled.reset_index(drop=True)

    df_shuffled.to_csv(TARGET_ROOT + '/master-target_image_path_labels.csv')

    # Preprocess csv file for files and labels
    fns_all = df['img'].tolist()
    lbs_all = df['img_label'].tolist()
    lbl_count = df.img_label.value_counts()
    #print(lbl_count)

    # Dictionary to match specimen name with label
    lookup_specimen_name = dict(zip([1,0],df.img_label.unique()))
    #print(lookup_specimen_name.items())

    for index,lbl in enumerate(lbs_all):
        if lbl.startswith("[u'Copepod']"):
            lbs_all[index] = 0
        else:
            lbs_all[index] = 1

    # Add source path to each image
    src_path = '/data4/plankton_wi17/mpl/target_domain/image_sym'
    fns_all = [os.path.join(src_path,img_fn) for img_fn in fns_all]

    # Check if first image can be found to test for entire list
    if not os.path.exists(fns_all[0]):
        print("Does not exist")

    # Return output to get_fns()
    return fns_all, lbs_all, lookup_specimen_name

def get_fns():
    """
    key - string denoting train, val, or test
    target - integer denoting the class number of the group (class, order, species, etc) to detect
    """

    # Read image paths and labels from csv
    fns,lbs,lookup_specimen_name = read_fns_from_csv()

    # shuffle
    index = range(len(fns))
    random.shuffle(index)
    fns = [fns[i] for i in index]
    lbs = np.array([lbs[i] for i in index])

    return fns, lbs

def match_img_id():
    img_dict = {}
    with open(HOME+"/image_id.txt","r") as f:
        img_n_id = [line.split('\t') for line in f]
        for img in img_n_id:
            img[1] = img[1].rstrip('\n')
            img[0] = str (img[0]) + '.jpg'
            img_dict.update([img])
        if DEBUG:
            print(img_dict)
    f.close()
    return img_dict

def main():

    # Check if main path to images exists
    if not os.path.exists(GENERAL_IMAGES_PATH):
        print("DirNotFoundError: Main path to images not found")
        return

    #match_img_id()
    # Get file names and labels
    test1_fns, test1_lbs = get_fns()

    write_caffe_lmdb(test1_fns, test1_lbs, os.path.join(TARGET_ROOT,'target_fourhrs.LMDB'))

    #lmdb = 'test1.LMDB'
    #ath = '/data4/plankton_wi17/plankton/plankton_binary_classifiers/plankton_phytoplankton/code'
    #lmdb_to_read = os.path.join(path,lmdb)
   # check_caffe_lmdb(lmdb_to_read)

if __name__ == '__main__':
    #main()
    fns,lbs,lookup_specimen_name = read_fns_from_csv()
    '''
    1. GENERAL_PATH
    2. which files to make lmdb for
    3. path hierarchy
    '''
