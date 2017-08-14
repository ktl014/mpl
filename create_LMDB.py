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

# lab_google_root = '/data4/plankton_wi17/plankton/compare_google_lab/copepod_order/lab' # SVCL Project
plank_prorocentrum_root = '/data4/plankton_wi17/plankton/plankton_binary_classifiers/plankton_phytoplankton'
spcinsitu_root = '/data4/plankton_wi17/mpl/source_domain/spcinsitu'
classifier = 'insitu_finetune'
# TIME_STAMP = '2017-4-10' # Prorocentrum Project
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
    df = pd.read_csv(spcinsitu_root + '/dataset_tools/target_image_path_labels.csv') # Original image list from spc website

    # Shuffle dataframe
    df_shuffled = df.iloc[np.random.permutation(len(df))]
    df_shuffled = df_shuffled.reset_index(drop=True)
    if DEBUG:
        print(df_shuffled)
    df_shuffled.to_csv('master-target_image_path_labels.csv') # Shuffled version

    # Preprocess csv file for files and labels
    fns_all = df['img'].tolist()
    lbs_all = df['img_label'].tolist()
    lbl_count = df.img_label.value_counts()
    if DEBUG:
        print(lbl_count)

    # Replace server labels with integers
    # "[u'Copepod']" -> 0, "[]} -> 1
    for index, lbl in enumerate (lbs_all):
        if lbl.startswith ("[u'Copepod']"):
            lbs_all[index] = 0
        else:
            lbs_all[index] = 1
    lbs_all = np.asarray(lbs_all)

    # Add source path to each image
    src_path = spcinsitu_root + '/all_insitu_images'
    fns_all = [os.path.join(src_path,img_fn) for img_fn in fns_all]

    # Check if first image can be found to test for entire list
    if not os.path.exists(fns_all[0]):
        print("Does not exist")

    # Combine image id with image
    copepod_imgid = df["img_id"][df['img_label']=="[u'Copepod']"].tolist()
    non_copepod_imgid = df["img_id"][df['img_label']=="[]"].tolist()
    copepod = df["img"][df['img_label']=="[u'Copepod']"].tolist() # Copepod images
    non_copepod = df["img"][df['img_label'] == '[]'].tolist () # Non-Copepod images

    copepod_dict={"img":copepod,"img_id":copepod_imgid}
    non_copepod_dict={"img":non_copepod,"img_id":non_copepod_imgid}


    # Partition image list into training, validation, and testing

    train_fns, train_lbs, val_fns, val_lbs, test1_fns, test1_lbs = partition(copepod_dict,non_copepod_dict)
    #train_fns, train_lbs = partition(copepod,non_copepod)

    # Return output to get_fns()
    return train_fns, train_lbs, val_fns, val_lbs, test1_fns, test1_lbs

def partition(class_0_dict,class_1_dict):
    num_img_class0 = len(class_0_dict['img'])
    num_img_class1 = len(class_1_dict['img'])

    class_0 = class_0_dict['img']
    class_1 = class_1_dict['img']

    # Construct training set
    train0_img = class_0_dict['img'][:int (num_img_class0 * 0.7)]
    train0_id = class_0_dict['img_id'][:int (num_img_class0 * 0.7)]
    train_dict = {'img':train0_img,'img_id':train0_id}
    train1_img = class_1_dict['img'][:int (num_img_class1 * 0.7)]
    train1_id = class_1_dict['img_id'][:int (num_img_class1 * 0.7)]
    train_dict['img'] = train_dict['img'] + train1_img
    train_dict['img_id'] = train_dict['img_id'] + train1_id

    # Set training set
    train_set = train_dict['img']
    train_id = train_dict['img_id']
    train_lbs = [0]*len(train0_img) + [1]*len(train1_img)

    # Construct validation set
    val0_img = class_0_dict['img'][int (num_img_class0 * 0.7):int (num_img_class0 * 0.8)]
    val0_id = class_0_dict['img_id'][int (num_img_class0 * 0.7):int (num_img_class0 * 0.8)]
    val_dict = {'img':val0_img,'img_id':val0_id}
    val1_img = class_1_dict['img'][int (num_img_class0 * 0.7):int (num_img_class0 * 0.8)]
    val1_id = class_1_dict['img_id'][int (num_img_class0 * 0.7):int (num_img_class0 * 0.8)]
    val_dict['img'] = val_dict['img'] + val1_img
    val_dict['img_id'] = val_dict['img_id'] + val1_id

    # Set validation set
    val_set = val_dict['img']
    val_id = val_dict['img_id']
    val_lbs = [0]*len(val0_img)+ [1]*len(val1_img)

    # Construct test set
    test0_img = class_0_dict['img'][int (num_img_class0 * 0.8):]
    test0_id = class_0_dict['img_id'][int (num_img_class0 * 0.8):]
    test_dict = {'img':test0_img,'img_id':test0_id}
    test1_img = class_1_dict['img'][int (num_img_class0 * 0.8):]
    test1_id = class_1_dict['img_id'][int (num_img_class0 * 0.8):]
    test_dict['img'] = test_dict['img'] + test1_img
    test_dict['img_id'] = test_dict['img_id'] + test1_id

    # Set test set
    test_set = test_dict['img']
    test_id = test_dict['img_id']
    test_lbs = [0]*len(test0_img) + [1]*len(test1_img)

    train_set,train_lbs = randomize_writepaths(train_set,train_id,train_lbs,'train')
    val_set,val_lbs = randomize_writepaths(val_set,val_id,val_lbs,'val')
    test_set,test_lbs = randomize_writepaths(test_set,test_id,test_lbs,'test')


    return train_set, train_lbs, val_set, val_lbs, test_set, test_lbs

def randomize_writepaths(fns,id,lbs,key):

    # Sanity check after partitioning
    if len(fns) != len(lbs):
        raise ValueError (key + " Number of files and labels do not match. {} vs {}".format(len(fns),len(lbs)))

    index = range(len(fns))
    random.shuffle(index)
    src_path = spcinsitu_root + '/all_insitu_images'
    #fns = [os.path.join(src_path,str(fns[i])) for i in index]
    for i in index:
        fns[i] = os.path.join(src_path,str(fns[i]))
    id = [id[i] for i in index]
    lbs = np.array([lbs[i] for i in index])
    if DEBUG:
        print(lbs.reshape(len(lbs),1))
    path_txt = open (spcinsitu_root + '/' + classifier + '/code/' + key + '.txt', 'w')
    nSmpl = lbs.size
    for i in range (nSmpl):
        path_txt.write (str(fns[i]) + ';' + str(id[i]) + ";" + str (lbs[i]) + '\n')
    path_txt.close ()
    return fns,lbs

def match_img_id():
    '''
    Scrapped function. No longer need
    :return: 
    '''
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
    if not os.path.exists(spcinsitu_root+'/all_insitu_images'):
        raise ValueError ('Image dir not found')

    # Get file names and labels
    train_fns, train_lbs, val_fns, val_lbs, test1_fns, test1_lbs = read_fns_from_csv()
    print("Files received")

    write_caffe_lmdb(train_fns, train_lbs, os.path.join(spcinsitu_root,classifier,'code','train.LMDB'))
    write_caffe_lmdb(val_fns, val_lbs, os.path.join(spcinsitu_root,classifier,'code','val.LMDB'))
    write_caffe_lmdb(test1_fns, test1_lbs, os.path.join(spcinsitu_root,classifier,'code','test1.LMDB'))

    #lmdb = 'test1.LMDB'
    #ath = '/data4/plankton_wi17/plankton/plankton_binary_classifiers/plankton_phytoplankton/code'
    #lmdb_to_read = os.path.join(path,lmdb)
   # check_caffe_lmdb(lmdb_to_read)

if __name__ == '__main__':
    main()
    '''
    1. GENERAL_PATH
    2. which files to make lmdb for
    3. path hierarchy
    '''
