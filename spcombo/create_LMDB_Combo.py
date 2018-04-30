"""Description
Create LMDB files from images located in image dir made from create_dataset.py script
Combo - Version
Author: Kevin Le
"""
import os
import random
import sys
import lmdb
import caffe
import numpy as np
import glob
import shutil
import cv2

DEBUG = False
spcombo_root = '/data4/plankton_wi17/mpl/source_domain/spcombo'
DATASETVERSION = 'V1b'

def write_caffe_lmdb(img_fns, labels, lmdb_fn):
    '''
    Writes LMDB file from image files and labels
    :param img_fns: list of shuffled image files
    :param labels: array of labels
    :param lmdb_fn: str of LMDB file name
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

        img = caffe.io.load_image(img_fns[i], color=False)   # Read image
        # img = caffe.io.resize_image(img, np.array([256, 256]))     # Resize to 256
        img = (img*255).astype(np.uint8)        # [0,1]->[0,255]
        img = np.dstack((img, img, img))        # Concatenate grayscale along 3rd dimension
        img = aspect_resize(img)       # Preserve aspect ratio and resize to 256
        if i == 1:
            cv2.imwrite(os.getcwd() + '/resized_img.png',img)
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
            print 'Samples saved:', i, '/', nSmpl
            sys.stdout.flush()
    return


def aspect_resize(im):
    '''
    Preserve aspect ratio and resizes the image
    :param im: data array of image rescaled from 0->255
    :return: resized image
    '''
    ii = 256
    mm = [int (np.median (im[0, :, :])), int (np.median (im[1, :, :])), int (np.median (im[2, :, :]))]
    cen = np.floor (np.array ((ii, ii)) / 2.0).astype ('int')  # Center of the image
    dim = im.shape[0:2]
    if DEBUG:
        print "median {}".format (mm)
        print "ROC {}".format (cen)
        print "img dim {}".format (dim)
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
            print 'before resize {}'.format (im.shape)
        im = cv2.resize (im, dim)
        half = np.floor (np.array (im.shape[0:2]) / 2.0).astype ('int')
        if DEBUG:
            print 'after resize {}'.format (im.shape)

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
        print 'aspect_resize: {}'.format (res.shape)
    return res

def get_fns(key, stats_f, partition, dataset):
    '''
    Gets image files from image dir
    :param key: str like train, val, or test
    :param stats_f: text file object to write dataset statistics to
    :param partition: list of partition value for the source domain dataset
    :param dataset: str of the dataset name
    :return: shuffled list of image files and array of labels
    '''
    subset_name = dataset
    fns = []
    lbs = []

    # Assign percentage based on key
    key_partition_dict = {'train': ['000','070'], 'val': ['070','080'], 'test': ['080','100']}
    for i in key_partition_dict.iterkeys():
        if key == i:
            key_partition = key_partition_dict[i]

    # Initialize paths to each source domain
    source_domains = ['spcbench','spcinsitu']
    img_root = spcombo_root + '/combo_images_exp/{}/'.format(DATASETVERSION)
    source_domains = [os.path.join(img_root,domain) for domain in source_domains] # Join path with source domain

    for sourceDomain_path in source_domains:
        domain_name = os.path.basename(sourceDomain_path)

        # Set partition based on source domain
        if domain_name == "spcbench":
            percentage = partition[0] # First value in partition list
        elif domain_name == "spcinsitu":
            percentage = partition[1] # Second value in partition list
        if DEBUG:
            print(sourceDomain_path + ' {} : {}%'.format(key,percentage))
        stats_f.write("srcdomain: " + str(sourceDomain_path.split('/')[7]) + '\n') # Source Domain heading in dataset stats text file

        # Hierarchy to retrieve images
        # source domain --> classes --> images
        class_list = glob.glob(os.path.join(sourceDomain_path, "*"))            # Grab list of classes

        for class_i in class_list:
            class_label = int(os.path.basename(class_i).replace("class",""))    # Grab label from path

            # Excludes bench non-copepod images and uses all Insitu non-copepod images
            if 'noise' in subset_name:
                if domain_name == "spcinsitu" and class_label == 1:
                    percentage = 100
                elif domain_name == "spcbench" and class_label == 1:
                    percentage = 0

            # Grab list of images based on dataset version
            if DATASETVERSION == 'V1b':
                img_list = open(class_i+'/data{}.txt'.format(class_label)).read().splitlines()
            else:
                img_list = glob.glob(os.path.join(class_i, "*"))

            # Shuffle retrieved image list before partitioning into train, val, and test sets
            if DEBUG:
                print 'train image 0 before shuffling: ', img_list[0]
            image_index = range(len(img_list))
            random.shuffle(image_index)
            img_list = [img_list[i] for i in image_index]
            if DEBUG:
                print 'train image 0 after shuffling: ', img_list[0]

            # Partition dataset based on source domain and key (i.e. domain=spcbench, key=train)
            total_num_img = int(len(img_list) * float(percentage) / 100) # Partition based off assigned partitions of source domain
            partitioned_img_list = img_list[0:total_num_img]             # List of images after partition

            # Initalize start and end indices to partition through list slicing
            start_num = int(int(key_partition[0]) * int(total_num_img) / 100)
            end_num = int(int(key_partition[1]) * int(total_num_img) / 100)
            if DEBUG:
                print "number of images: {}".format(end_num-start_num)
            stats_f.write("\t" + "Class {}: ".format(class_label) + "{}".format(end_num-start_num) + '\n')

            # Add to returning files and labels
            fns += partitioned_img_list[start_num:end_num]
            lbs += [class_label]*(end_num-start_num)

            # Reset partitions back to normal
            if os.path.basename (sourceDomain_path) == "spcbench":
                percentage = partition[0]
            elif os.path.basename (sourceDomain_path) == "spcinsitu":
                percentage = partition[1]

    # Initialize destination path to write train, val, and test text files to
    dest_path = os.path.join(spcombo_root + '/combo_finetune',subset_name.split('_')[0],subset_name,'code')  # Grabs current dir to write dest path
    fns, lbs = randomize_writepaths(fns,lbs,key,dest_path)
    return fns, lbs

def randomize_writepaths(fns, lbs, key, dest_path):
    '''
    Randomize images and labels
    :param fns: list of file names
    :param lbs: list of labels
    :param key: str like train, val, or text
    :param dest_path: path to write text file of all files and labels randomized
    :return: shuffled files and labels
    '''

    # Shuffle fns and labels
    index = range (len (fns))
    random.shuffle (index)
    fns = [fns[i] for i in index]
    lbs = np.array ([lbs[i] for i in index])

    # Write img and labels to text file
    with open (dest_path + '/{}.txt'.format (key), "w") as f:
        for i in range (len (fns)):
            f.write (str (fns[i]) + ' ' + str (lbs[i]) + '\n')
    f.close ()
    return fns, lbs

def main():
    if not os.path.exists(spcombo_root+'/combo_images_exp'):
        raise ValueError ('Image dir not found')

    # Datasets to be created based off each sub-partition list
    # ['XX','YY'] -> Bench Partition, Insitu Partition
    # including 'noise' in key name will exclude all bench non-copepod images
    datasets = {
          # "benchv1b-noise100": [['100','01'],['100', '05'],['100', '10'],['100', '15'],['100', '20'], ['100', '40'],
          #                    ['100', '50'], ['100', '60'], ['100', '80']],
          #
          # "insituv1b-noise100":[['0.25', '100'], ['0.5', '100'], ['1.5', '100'], ['002', '100'], ['7.5', '100'],['12.5', '100'], ['14', '100'],
          #                    ['001','100'],['005','100'],['010','100'],['015','100'],['020','100'],['040','100'],['050','100'],['060','100'],
          #                    ['080','100']],

          "allv1b-noise100": [['100', '100']]
    }

    # Iterate over each dataset key
    # order: dataset -> partition list -> percentage
    for dataset in datasets.iteritems():
        for index,subset in enumerate(dataset[1]):  # Iterate over each partition list
            partition = subset                      # For instance -> ['020','100']
            subset_name = dataset[0] + "_{}-{}".format(int(partition[0]),str(partition[1]))     # For instance -> bench-noise100_XX-YY
            print subset_name

            # Create dest path to output dataset txt and LMDB files
            spcombo = spcombo_root + '/combo_finetune'
            dest_path = os.path.join(spcombo,dataset[0],subset_name,'code') # creates folders in current dir of create_LMDB script
            if not os.path.exists(dest_path):
                assert "Check Path"

            # Copy python scripts to each dataset from github repository
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
                gitRoot = '/data4/plankton_wi17/mpl/mpl_git'
                evalScript = gitRoot + '/eval.py'
                trainScript = gitRoot +  '/train_alexnet.py'
                shutil.copy(evalScript,dest_path+'/eval.py')
                shutil.copy(trainScript,dest_path+'/train_alexnet.py')
                print 'Copied eval and train script'

            # Set up first experiment directory
            if not os.path.exists(os.path.join(os.getcwd(),dataset[0],subset_name,'exp1')):
                os.makedirs(os.path.join(os.getcwd(),dataset[0],subset_name,'exp1'))

            # Copy CaffeNet files to each dataset directory
            caffe_fldr = '/data4/plankton_wi17/mpl/mpl_git/caffenet'
            if not os.path.exists(dest_path+'/caffenet'):
                print 'Copying caffe files'
                shutil.copytree(caffe_fldr,dest_path+'/caffenet')

            existingLMDBVersion = 'V1a'
            for key in ['train', 'val', 'test1']:
                lmdb_path = dest_path + '/{}.LMDB'.format(key)
                if os.path.exists(lmdb_path):
                    dest_lmdb_path = dest_path + '/{}LMDB'.format(existingLMDBVersion)
                    if not os.path.exists(dest_lmdb_path):
                        os.makedirs(dest_lmdb_path)
                    shutil.move(lmdb_path, dest_lmdb_path + '/{}.LMDB'.format(key))

            # # Write stats for each dataset
            with open(dest_path + "/{}_stats.txt".format(subset_name), "w") as stats_f:
                # Write stats file for each dataset

                # Get image files and labels for each set
                train_fns, train_lbs = get_fns ('train', stats_f, partition, subset_name)
                stats_f.write("Total {} files: {}".format('train',len(train_fns)) + '\n\n')
                val_fns, val_lbs = get_fns ('val', stats_f, partition, subset_name)
                stats_f.write("Total {} files: {}".format('val',len(val_fns)) + '\n\n')
                test1_fns, test1_lbs = get_fns ('test', stats_f, partition, subset_name)
                stats_f.write("Total {} files: {}".format('test',len(test1_fns)) + '\n\n')

            stats_f.close()

            # Write LMDB Files
            write_caffe_lmdb (train_fns, train_lbs, os.path.join (dest_path, 'train.LMDB'))
            write_caffe_lmdb (val_fns, val_lbs, os.path.join (dest_path, 'val.LMDB'))
            write_caffe_lmdb (test1_fns, test1_lbs, os.path.join (dest_path, 'test1.LMDB'))

if __name__ == '__main__':
    main()

