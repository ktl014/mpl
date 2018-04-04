"""Description
"""
from __future__ import print_function
import os
import sys
import argparse
import caffe
import timeit
import shutil
EXP_NUM = 'exp3'

def train(srcPath):
    solver_fn = srcPath + '/caffenet/solver_' + EXP_NUM + '.prototxt'
    if not os.path.exists(solver_fn):
        previousEXP = 'exp' + str(int(EXP_NUM[3:])-1)
        if os.path.exists(srcPath + '/caffenet/solver_' + previousEXP + '.prototxt'):
            print('copied new solver file for ' + EXP_NUM)
            shutil.copy(srcPath + '/caffenet/solver_' + previousEXP + '.prototxt', solver_fn)

    # Set which GPU to use
    gpu_id = 1
    caffe.set_device(gpu_id)
    # Set GPU mode
    caffe.set_mode_gpu()

    # Set # of test iterations
    test_iters = 1501

    model_pretrained = srcPath + '/caffenet/bvlc_reference_caffenet.caffemodel'

    # Load solver (Default is the SGD solver)
    solver = caffe.SGDSolver(solver_fn)     # Prep solver
    solver.net.copy_from(model_pretrained)


    # Get layer types
    layer_types = [ll.type for ll in solver.net.layers]
    #print(layer_types)

    # Get the indices of layers that have weights in them
    weight_layer_idx = [idx for idx, l in enumerate(layer_types) if 'Convolution' in l or 'InnerProduct' in l]
    #print(weight_layer_idx)

    t1 = timeit.default_timer() # Start timer
    for i in range(test_iters):

        # Launch one step of the gradient descent
        # (forward propagation, backward propagation, and update of net params given gradients)
        solver.step(1)

    solver.net.save(srcPath + '/model_' + EXP_NUM + '.caffemodel')

    t2 = timeit.default_timer() # End timer
    print("Training Time: {}".format(t2-t1))

def trainmultiple_Models():
    mainSRC = '/data4/plankton_wi17/mpl/source_domain/spcombo/combo_finetune/bench-noise100'
    datasets = {
        "bench-noise100": [ ['100', '10'], ['100', '15'], ['100', '20'], ['100', '40'],
                           ['100', '50'], ['100', '60'], ['100', '80']], } # ['100', '01'], ['100', '05'],
    for dataset in datasets.iteritems():
        for subset in dataset[1]:
            subsetName = dataset[0] + "_{}-{}".format (int (subset[0]),
                                                    str (subset[1]))  # For instance -> bench-noise100_XX-YY
            srcPath = os.path.join(mainSRC, subsetName, 'code')
            print =('training ' + subsetName)
            train(srcPath)

    # Select which folder to train

    # Navigate to each folder's solver fn
if __name__ == '__main__':
    srcPath = os.getcwd()
    train(srcPath)
    trainmultiple_Models()
