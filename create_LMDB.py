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
# import basic_augment

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

def read_fns_from_csv():

    # Open csv file and convert files and labels to list
    df = pd.read_csv('ogspc-fourhr_target_image_path_labels.csv')

    # Shuffle dataframe
    df_shuffled = df.iloc[np.random.permutation(len(df))]
    df_shuffled = df_shuffled.reset_index(drop=True)

    df_shuffled.to_csv(TARGET_ROOT + '/master1-fourhr_target_image_path_labels.csv')

    # Preprocess csv file for files and labels
    fns_all = df['img'].tolist()
    lbs_all = df['img_label'].tolist()
    imgid_all = df['img_id'].tolist()
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
    lbs_all = np.asarray(lbs_all)

    # Add source path to each image
    src_path = '/data4/plankton_wi17/mpl/target_domain/image_sym'
    fns_all = [os.path.join(src_path,img_fn) for img_fn in fns_all]

    # Check if first image can be found to test for entire list
    if not os.path.exists(fns_all[0]):
        print("Does not exist")

    # Combine image id with image
    # copepod_imgid = df["img_id"][df['img_label']=="[u'Copepod']"].tolist()
    # non_copepod_imgid = df["img_id"][df['img_label']=="[]"].tolist()
    copepod = df["img"][df['img_label']=="[u'Copepod']"].tolist() # Copepod images
    non_copepod = df["img"][df['img_label'] == '[]'].tolist () # Non-Copepod images

    # copepod_dict={"img":copepod,"img_id":copepod_imgid}
    # non_copepod_dict={"img":non_copepod,"img_id":non_copepod_imgid}


    # Partition image list into training, validation, and testing

    #train_fns, train_lbs, val_fns, val_lbs, test1_fns, test1_lbs = partition(copepod_dict,non_copepod_dict)
    train_fns, train_lbs, val_fns, val_lbs, test1_fns, test1_lbs = partition(copepod,non_copepod)

    # Return output to get_fns()
    return train_fns, train_lbs, val_fns, val_lbs, test1_fns, test1_lbs

def partition(class_0,class_1):
    f = open("stats_1.txt","w")
    num_img_class0 = len(class_0)
    num_img_class1 = len(class_1)
    print("Class 0 {}".format(num_img_class0))
    print("Class 1 {}".format(num_img_class1))


    # Construct training set
    train0_img = class_0[:int (num_img_class0 * 0.7)]
    train1_img = class_1[:int (num_img_class1 * 0.7)]

    # Set training set
    train_set = train0_img + train1_img
    train_lbs = [0]*len(train0_img) + [1]*len(train1_img)

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

    train_set,train_lbs = randomize_writepaths(train_set, train_lbs, 'train')
    val_set,val_lbs = randomize_writepaths(val_set, val_lbs, 'val')
    test_set,test_lbs = randomize_writepaths(test_set, test_lbs, 'test')

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

def partition_1(class_0_dict,class_1_dict):
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

def randomize_writepaths(fns,lbs,key):

    # Sanity check after partitioning
    if len(fns) != len(lbs):
        raise ValueError (key + " Number of files and labels do not match. {} vs {}".format(len(fns),len(lbs)))

    index = range(len(fns))
    random.shuffle(index)
    src_path = spcinsitu_root + '/all_insitu_images'
    #fns = [os.path.join(src_path,str(fns[i])) for i in index]
    for i in index:
        fns[i] = os.path.join(src_path,str(fns[i]))
    lbs = np.array([lbs[i] for i in index])
    if DEBUG:
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

def main():

    # Check if main path to images exists
    if not os.path.exists(GENERAL_IMAGES_PATH):
        print("DirNotFoundError: Main path to images not found")
        return

    #match_img_id()
    # Get file names and labels
    test1_fns, test1_lbs = read_fns_from_csv()
    with open("test.txt",'w') as f:
        for i in range(len(test1_fns)):
            f.write(str(test1_fns[i] + " " + str(test1_lbs[i]) + '\n'))
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

    #write_caffe_lmdb(test1_fns, test1_lbs, os.path.join(TARGET_ROOT,'target_fourhrs.LMDB'))
=======
    train_fns, train_lbs, val_fns, val_lbs, test1_fns, test1_lbs = read_fns_from_csv()
    print("train {} images".format(len(train_fns)))
    print("val {} images".format(len(val_fns)))
    print("test {} images".format(len(test1_fns)))


    #train_fns, train_lbs = combine_target2train_dataset()
    print("Files received")

    #write_caffe_lmdb(train_fns, train_lbs, os.path.join(spcinsitu_root,classifier,'code','train_combined-w-target.LMDB'))
    #write_caffe_lmdb(train_fns, train_lbs, os.path.join(spcinsitu_root,classifier,'code','train.LMDB'))
    #write_caffe_lmdb(val_fns, val_lbs, os.path.join(spcinsitu_root,classifier,'code','val.LMDB'))
    #write_caffe_lmdb(test1_fns, test1_lbs, os.path.join(spcinsitu_root,classifier,'code','test1.LMDB'))
>>>>>>> dbbbb2372ef971ae091ec0bd1a7089efae9a9844

    #lmdb = 'test1.LMDB'
    #ath = '/data4/plankton_wi17/plankton/plankton_binary_classifiers/plankton_phytoplankton/code'
    #lmdb_to_read = os.path.join(path,lmdb)
   # check_caffe_lmdb(lmdb_to_read)

if __name__ == '__main__':
    main()
<<<<<<< HEAD
    #fns,lbs,lookup_specimen_name = read_fns_from_csv()
=======
    #combine_target2train_dataset()
>>>>>>> dbbbb2372ef971ae091ec0bd1a7089efae9a9844
    '''
    1. GENERAL_PATH
    2. which files to make lmdb for
    3. path hierarchy
    '''
