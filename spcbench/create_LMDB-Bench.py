"""Description
Creates LMDB files for inputting images and labels into network
BENCH Version
"""
from __future__ import print_function
import os
import random
import sys
import lmdb
import caffe
import numpy as np
import glob
import cv2

CLASSIFIER = 'bench_from_finetune' # Change Classifier
GENERAL_CODE_PATH = '/data4/plankton_wi17/mpl/source_domain/spcbench/{}/code'.format(CLASSIFIER)
GENERAL_IMAGES_PATH = '/data4/plankton_wi17/mpl/source_domain/spcbench/bench_images_exp'
DEBUG = False


def write_caffe_lmdb(img_fns, labels, lmdb_fn):
    '''
    Writes LMDB file to input into Caffenetwork
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
        img = (img*255).astype(np.uint8)        # [0,1]->[0,255]
        img = aspect_resize(img)                # Preserve aspect ratio and resize to 256
        img = img[:, :, (2, 1, 0)]              # RGB->BGR
        img = np.transpose(img, (2, 0, 1))      # [X,Y,C]->[C,X,Y]
        if DEBUG:
            print ('Img Max, Img Min, Img Mean, Img Shape')
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


def get_fns(key, target_lb, stats_f):
    '''
    Grab image files and labels needed for training and testing
    :param key: string denoting train, val, or test
    :param target_lb: integer denoting the class number of the group (class, order, species, etc) to detect
    :param stats_f: text file to compile all image statistics together
    :return: img_fns and labels to process into LMDB
    '''

    class_paths_all = glob.glob(os.path.join(GENERAL_IMAGES_PATH,'class*'))
    if DEBUG:
        print ('List of classes found in class dir:')
        print (class_paths_all[0:5])
    fns = []
    lbs = []
    # directory hierarchy:
    # images -> class** -> subclass** -> train/val/test
    # class is family; subclass is specimen
    specimen_count_class0 = 0
    specimen_count_class1 = 0
    target_new_lb = 0
    others_new_lb = 1
    num_class0 = 0
    num_class1 = 0
    for class_path in class_paths_all:

        # get class label
        class_name = class_path.split('/')[-1]
        class_label = int(class_name[5:])

        # get all subclasses in class dir
        subclass_paths_all = glob.glob(os.path.join(class_path,'subclass*'))

        # Identify all classes in directory as Class01 (Non-Copepod) except designate Class00 as Copepod Class
        if class_label == target_lb:
            label = target_new_lb
            num_class0 += 1
            specimen_count_class0 += len(subclass_paths_all)
        else:
            label = others_new_lb
            specimen_count_class1 += len(subclass_paths_all)
            num_class1 += 1

        for subclass_path in subclass_paths_all:
            if DEBUG:
                print (subclass_path.split('/')[-1])

            # get all images in subclass dir
            class_fns = [os.path.join(subclass_path, key, fn)
                         for fn in os.listdir(os.path.join(subclass_path, key)) if fn.endswith('.png')]
            fns += class_fns
            lbs += [label]*len(class_fns)

    # check if binary separation is successful
    num_class0_fles = lbs.count(0)
    num_class1_fles = lbs.count(1)

    # print training, val, and test dataset stats
    create_stats_f(stats_f,key,fns,lbs,num_class0,num_class1,specimen_count_class0,specimen_count_class1,num_class0_fles,num_class1_fles)

    # shuffle
    index = range(len(fns))
    random.shuffle(index)
    fns = [fns[i] for i in index]
    lbs = np.array([lbs[i] for i in index])

    # write paths and labels to txt files
    path_txt = open('Image_paths_labels_'+key+'.txt','w')
    nSmpl = lbs.size
    for i in range(nSmpl):
        path_txt.write(fns[i]+' '+str(lbs[i])+'\n')
    path_txt.close()
    return fns, lbs


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

def create_stats_f(f,key,fns,lbs,num_class0,num_class1,specimen_count_class0,specimen_count_class1,num_class0_fles,num_class1_fles):
    '''
    Collect statistics of each dataset into one text file
    all param are self-explanatory
    :return: dataset stats text file
    '''
    f.write('key: ' + key + '\n')
    # f.write('# specimens:', specimen_count)
    f.write('# images: ' + str(len(fns)) + '\n')
    f.write('# labels: ' + str(len(lbs))+ '\n')
    f.write('# target class0: ' + str(num_class0)+ '\n')
    f.write('# other class1: ' + str(num_class1)+ '\n')
    f.write('# specimens class0: ' + str(specimen_count_class0) +'\n')
    f.write('# specimens class1: ' + str(specimen_count_class1) + '\n')
    f.write('# target class0 files: ' + str(num_class0_fles)+ '\n')
    f.write('# target class1 files: ' + str(num_class1_fles)+ '\n')
    f.write('\n')
    print(str(key) + " stats.txt successfuly saved to " + str(GENERAL_CODE_PATH))

def main():

    # Check if main path to images exists
    if not os.path.exists(GENERAL_IMAGES_PATH):
        print("DirNotFoundError: Main path to images not found")
        return


    # get file names and lables
    target_lb = 1
    with open("aspect_stats.txt",'w') as f:
        train_fns, train_lbs = get_fns('train', target_lb, f)
        val_fns, val_lbs = get_fns('val', target_lb, f)
        test1_fns, test1_lbs = get_fns('test1', target_lb, f) # same specimen
    f.close()

    # write lmdb files
    write_caffe_lmdb(train_fns, train_lbs, 'train.LMDB')
    write_caffe_lmdb(val_fns, val_lbs, 'val.LMDB')
    write_caffe_lmdb(test1_fns, test1_lbs, 'test1.LMDB')


if __name__ == '__main__':
    main()
    '''
    1. GENERAL_PATH
    2. which files to make lmdb for
    3. path hierarchy
    '''
