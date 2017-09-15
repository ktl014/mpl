"""Description
"""
from __future__ import print_function
import os
import sys
import argparse
import caffe
import timeit

# Specify which exp_num to use. A new solver.prototxt must be made to match with desired exp_num
EXP_NUM = 'exp2'

def main():
    solver_fn = 'caffenet/solver_' + EXP_NUM + '.prototxt'
    if not os.path.exists(solver_fn):
        raise ValueError('Solver_fn not found or does not exist')

    # Set which GPU to use
    gpu_id = 1
    caffe.set_device(gpu_id)
    # Set GPU mode
    caffe.set_mode_gpu()

    # Set # of test iterations
    test_iters = 3001

    model_pretrained = 'caffenet/bvlc_reference_caffenet.caffemodel'

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

    solver.net.save('model_' + EXP_NUM + '.caffemodel')

    t2 = timeit.default_timer() # End timer
    print("Training Time: {}".format(t2-t1))
if __name__ == '__main__':
    main()
