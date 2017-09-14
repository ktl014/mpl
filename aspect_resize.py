#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:09:28 2017

@author: eric
"""
import os
import random
import sys
import lmdb
import caffe
import numpy as np
import cv2
import shutil
#from skimage import morphology, restoration
#from skimage.filters import scharr, gaussian
#from scipy import ndimage
DEBUG = False
def aspect_resize(im):
    ii = 256
    mm = [int(np.median(im[0,:,:])), int(np.median(im[1,:,:])), int(np.median(im[2,:,:]))]
    cen = np.floor(np.array((ii,ii))/2.0).astype('int') # Center of the image
    dim = im.shape[0:2]
    if DEBUG:
        print "median {}".format(mm)
        print "ROC {}".format(cen)
        print "img dim {}".format(dim)
        # exit(0)

    if dim[0] != dim[1]:
        # get the largest dimension
        large_dim = max(dim)
        
        # ratio between the large dimension and required dimension
        rat = float(ii)/large_dim
        
        # get the smaller dimension that maintains the aspect ratio
        small_dim = int(min(dim)*rat)
        
        # get the indicies of the large and small dimensions
        large_ind = dim.index(max(dim))
        small_ind = dim.index(min(dim))
        dim = list(dim)
        
        # the dimension assigment may seem weird cause of how python indexes images
        dim[small_ind] = ii
        dim[large_ind] = small_dim
        dim = tuple(dim)
        if DEBUG:
            print 'before resize {}'.format(im.shape)
        im = cv2.resize(im,dim)
        half = np.floor(np.array(im.shape[0:2])/2.0).astype('int')
        if DEBUG:
            print 'after resize {}'.format(im.shape)

        # make an empty array, and place the new image in the middle
        res = np.zeros((ii,ii,3), dtype='uint8')
        res[:,:,0] = mm[0]
        res[:,:,1] = mm[1]
        res[:,:,2] = mm[2]
        
        if large_ind == 1:
            test = res[cen[0]-half[0]:cen[0]+half[0], cen[1]-half[1]:cen[1]+half[1]+1]
            if test.shape != im.shape:
                res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]+1] = im
            else:
                res[cen[0]-half[0]:cen[0]+half[0], cen[1]-half[1]:cen[1]+half[1]+1] = im
        else:
            test = res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]]
            if test.shape != im.shape:
                res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]+1] = im
            else:
                res[cen[0]-half[0]:cen[0]+half[0]+1, cen[1]-half[1]:cen[1]+half[1]] = im
    else:
        res = cv2.resize(im,(ii,ii))
        half = np.floor(np.array(im.shape[0:2])/2.0).astype('int')

    if DEBUG:
        print 'aspect_resize: {}'.format(res.shape)
    return res

def write_caffe_lmdb_tst(img_fns, labels, lmdb_fn):
    if not os.path.exists(lmdb_fn) and not DEBUG:
        raise ValueError(lmdb_fn + ' already exists.')

    nSmpl = labels.size
    map_size = nSmpl*3*256*256*8*1.5
    env_img = lmdb.open(lmdb_fn, map_size=map_size)
    print('Generating dataset lmdb: '+lmdb_fn)
    for i in range(nSmpl):
        # Write image datum
        datum = caffe.proto.caffe_pb2.Datum()

        if DEBUG:
            print "datatype before caffe load {}".format(cv2.imread(img_fns[i]).dtype)
        img = caffe.io.load_image(img_fns[i])   # Read image
        if DEBUG:
            print "datatype after caffe load {}".format(img.dtype)

        img = caffe.io.resize_image(img, np.array([256, 256]))     # Resize to 256
        img = (img*255).astype(np.uint8)        # [0,1]->[0,255]

        # img = aspect_resize(img)
        if DEBUG: # Check image after aspect is preserved and img is resized
            print "tst {}".format(img.shape)
            cv2.imwrite('/data4/plankton_wi17/mpl/source_domain/spcinsitu/dataset_tools/img_og.jpg',img)
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

def main():
    test_img = '/data4/plankton_wi17/mpl/source_domain/spcinsitu/all_insitu_images/SPC2-1441562601-021228-001-2316-2508-120-152.jpg'
    test_img2 = '/data4/plankton_wi17/mpl/source_domain/spcinsitu/all_insitu_images/SPC2-1441562679-021695-001-1720-1800-56-128.jpg'
    im = cv2.imread(test_img)
    # cv2.imwrite ('/data4/plankton_wi17/mpl/source_domain/spcinsitu/dataset_tools/test_img.jpg', im)
    dest_img = os.path.join(os.getcwd(),os.path.basename(test_img))
    # shutil.copy(test_img,dest_img)
    #im = aspect_resize(im)

    fns = [test_img,test_img2]
    lbl = np.asarray([0])
    write_caffe_lmdb_tst(fns,lbl,"test_aug.LMDB")
if __name__ == "__main__":
    main()