from __future__ import print_function
import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import *
import caffe
import lmdb
import pandas as pd
import os
import timeit
import csv
import sys
import itertools

# Specify which source domain & classifier will be used for evaluation
HOME = '/data4/plankton_wi17/mpl/source_domain'
TARGET = True                                       # Set to True to use trained model to test on target dataset
SRC_DOMAIN = 'spcombo'                                  # Source domain
CLASSIFIER_NAME = 'combo_finetune'                       # Name of classifier
EXP_NUM = 'exp1'                                    # Select which expNum model to use for evaluation
model = 'model_' + EXP_NUM + '.caffemodel'
OUTROOT = os.path.join(HOME, SRC_DOMAIN, CLASSIFIER_NAME)   # Sets up source/destination path for input and output files
if SRC_DOMAIN == 'spcombo':                         # Changes file paths for SPCombo SRC_DOMAIN
    dataset = 'insitu-noise100'
    datasetDegree = dataset + '_100-15'
    OUTROOT = os.path.join(OUTROOT, dataset, datasetDegree)


def main(test_data, num_class):
    t1 = timeit.default_timer()  # Start timer

    # Initialize paths to input files
    inputfile_dir = OUTROOT + '/code'  # Main directory for input files (lmdb, caffe prototxt, etc)
    lmdbfile = inputfile_dir + '/{}'.format(test_data)  # Sets up path to desired LMDB

    # Pulls target LMDB file if testing target dataset
    if TARGET:
        lmdbfile = '/data4/plankton_wi17/mpl/target_domain/aspect_target_fourhrs.LMDB'

    # Load LMDB file
    images, labels = load_lmdb(lmdbfile)

    # Set to GPU mode
    gpu_id = 1
    caffe.set_mode_gpu()
    # caffe.set_device(gpu_id)

    # Create path to deploy protoxt and weights
    deploy_proto = inputfile_dir + '/caffenet/deploy.prototxt'
    trained_weights = inputfile_dir + '/{}'.format(model)
    print('Using {}'.format(trained_weights))

    # Check if files can be found
    if not os.path.exists(deploy_proto):
        raise ValueError(os.path.basename(deploy_proto) + " not found")
    elif not os.path.exists(trained_weights):
        raise ValueError(os.path.basename(trained_weights) + " not found")

    # Load net
    deploy = caffe.Net(deploy_proto, caffe.TEST, weights=trained_weights)
    probs = []
    nSmpl = len(images)

    # Set up input preprocessing
    for i in range(0,len(images),25):

        # Configure preprocessing
        batch = [prep_image(img) for img in images[i:i +25]]
        batch_size = len(batch)

        # Load image in the data layer
        deploy.blobs['data'].data[:batch_size] = batch

        # Begin forward propagation
        deploy.forward()

        # Compute output probability vector from each image
        probs.append(np.copy(deploy.blobs['prob'].data[:batch_size,:])) # Note np.copy. Otherwise, next forward() step will replace memory

        if i%1000 == 0:
            print('Samples computed:', i, '/',nSmpl)
            sys.stdout.flush()

    t2 = timeit.default_timer() # End timer
    print ('GPU Mode time to evaluate {}'.format (t2 - t1))

    print ('probs list length:', len (probs))
    print ('probs element type:', type (probs[0]))
    print (probs[0])

    # Convert list to array type
    probs = np.concatenate(probs, 0)
    print ('probs shape after concatenate:', probs.shape)
    print (probs[0,:], type(probs[0,0]))

    # Compute accuracy
    predictions = probs.argmax (1)
    gtruth = np.array (labels)
    total_accu = (predictions == gtruth).mean () * 100
    error_rate = (predictions != gtruth).mean() * 100
    print ('predictions shape:', predictions.shape)
    print (predictions[0:25])
    print ('Total Accuracy', total_accu)
    print ('Error Rate', error_rate)


    # Create array for confusion matrix with dimensions based on number of classes
    confusion_matrix_rawcount = np.zeros ((num_class, num_class))
    class_count = np.zeros((num_class,1)) # 1st col represents number of images per class

    # Create confusion matrix
    for t, p in zip (gtruth, predictions):
        class_count[t,0] += 1
        confusion_matrix_rawcount[t, p] += 1
    confusion_matrix_rate = np.zeros((num_class,num_class))
    for i in range(num_class):
        confusion_matrix_rate[i,:] = (confusion_matrix_rawcount[i,:])/class_count[i,0]*100
    confusion_matrix_rate = np.around(confusion_matrix_rate, decimals=4)

    # Calculate Precision Rate
    precision = (confusion_matrix_rawcount[0,0]/(confusion_matrix_rawcount[0,0]+confusion_matrix_rawcount[1,0]))*100 # TP / (FP+TP)
    print("Precision {}".format(precision))

    # Calculate Recall Rate
    recall = (confusion_matrix_rawcount[0,0]/(confusion_matrix_rawcount[0,0]+confusion_matrix_rawcount[0,1]))*100 # TP / (FN+TP)
    print("Recall {}".format(recall))

    if TARGET:
        # Normalized Accuracy
        normAccu = (confusion_matrix_rate[0,0]+confusion_matrix_rate[1,1])/2.00
        print("Normalized Accuracy {}".format(normAccu))

        # Normalized Error Rate
        normError = 100.00 - normAccu
        print("Normalized Error {}".format(normError))


    # Calculate side lobes
    S = np.sort (probs)
    S = S[::-1]
    confidence_level = [(S[i, 1] - S[i, 0]) / S[i, 1] for i in range (len (S))]
    confidence_level = np.asarray(confidence_level)
    avg_confidence = confidence_level.mean()*100
    print ("Confidence Level {}".format(avg_confidence))

    dest_path = os.path.join (OUTROOT, EXP_NUM)
    if TARGET:
        dest_path = dest_path.replace("source","target")
    if not os.path.exists (dest_path):
        os.makedirs (dest_path)

    # Write predictions to img path lbl txt/csv file
    if SRC_DOMAIN == "spcbench":
        write_pred2txt(predictions, probs, dest_path)
    else:
        write_pred2csv(predictions,probs, inputfile_dir, dest_path)

    results_filename = os.path.join (dest_path, CLASSIFIER_NAME + '-' + EXP_NUM + '_Results.csv')
    outfile = open (results_filename, 'wb')
    writer = csv.writer (outfile, delimiter=",")
    writer.writerow (['Binary Classifier: {}-{}'.format(CLASSIFIER_NAME, EXP_NUM)])
    writer.writerow (['Total Accuracy'])
    writer.writerow ([str (total_accu)])
    writer.writerow (['Error Rate'])
    writer.writerow ([str (error_rate)])
    if TARGET:
        writer.writerow (['Normalized Accuracy'])
        writer.writerow ([str (normAccu)])
        writer.writerow (['Normalized Error Rate'])
        writer.writerow ([str (normError)])
    writer.writerow (['Precision Rate'])
    writer.writerow ([str (precision)])
    writer.writerow (['Recall Rate'])
    writer.writerow ([str (recall)])
    writer.writerow (['Confidence Level'])
    writer.writerow ([str (avg_confidence)])
    writer.writerow (['Confusion Matrix (Raw Count):'])
    np.savetxt (outfile, confusion_matrix_rawcount,'%5.2f',delimiter=",")
    writer.writerow (['Confusion Matrix (Rate):'])
    np.savetxt (outfile, confusion_matrix_rate,'%5.2f', delimiter=",")
    print ('Print to', results_filename, 'file successful.')

    # Plot ROC Curve
    plot_roc_curve(gtruth, probs, CLASSIFIER_NAME, num_class, dest_path)

    # Plot Confusion Matrix
    classNames = ['copepod','non-copepod']
    plt.figure()
    plt.subplot(211)
    plot_confusion_matrix(confusion_matrix_rawcount,classes=classNames,title='Confusion Matrix (Raw Count)')
    plt.subplot(212)
    plot_confusion_matrix(confusion_matrix_rate,classes=classNames,title='Confusion Matrix (Rate)')
    cnf_plot_filename = os.path.join(dest_path,'cnf_matrix.png')
    plt.savefig(cnf_plot_filename)

def load_lmdb(fn):
    '''
    Load LMDB
    :param fn: filename of lmdb
    :return: images and labels
    '''
    print("Loading " + str(fn))
    env = lmdb.open(fn,readonly=True)
    datum = caffe.proto.caffe_pb2.Datum()
    with env.begin() as txn:
        cursor = txn.cursor()
        data,labels = [],[]
        for _,value in cursor:
            datum.ParseFromString(value)
            labels.append(datum.label)
            data.append(caffe.io.datum_to_array(datum).squeeze())
    env.close()
    print("LMDB successfully loaded")
    return data, labels

def prep_image(img):
    img = img.astype(float)[:, 14:241, 14:241] # center crop (img is in shape [C,X,Y])
    img -= np.array([104., 117., 123.]).reshape((3,1,1)) # demean (same as in trainval.prototxt
    return img

def plot_roc_curve(gt,prob,num_class, dest_path):
    '''
    Plot ROC Curve
    :param gt: groundtruth labels
    :param prob: probability vector
    :param num_class: number of classes
    :param dest_path: where to output file
    :return: 
    '''
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # Reformats gtruth array to have <num_class> columns and moves gtruth label to associated class column
    gtruth = np.zeros((len(gt),num_class))
    for i in range(len(gt)):
        if gt[i]==1:
            gtruth[i,1] = 1
        else:
            gtruth[i,0] = 1

    # Plots curve using sk-learn kit
    for i in range(num_class):
        fpr[i],tpr[i],_ = roc_curve(gtruth[:,i],prob[:,i])
        roc_auc[i] = auc (fpr[i], tpr[i])

    #fpr["micro"], tpr["micro"], _ = roc_curve (gtruth.ravel (), pred.ravel ())

    #fpr["micro"], tpr["micro"], _ = roc_curve (gtruth.ravel (), prob.ravel ())
    #roc_auc["micro"] = auc (fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    colors = ['aqua','darkorange']
    for i, color in zip(range(num_class),colors):
        plt.plot(fpr[i],tpr[i],color=color,lw=lw,label='ROC curve of class {0} (area = {1:0.2f})'.format(i,roc_auc[i]))
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Postive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Designate file path and name to save ROC Curve
    fig_filename = os.path.join(dest_path, CLASSIFIER_NAME + '-' + EXP_NUM + '_curve.png')
    plt.savefig(fig_filename)
    print (CLASSIFIER_NAME+'_curve.png saved.')

def write_pred2csv(predictions, probs, inputfile_dir, dest_path):
    '''
    Write predictions and confidence level for each image to csv file to upload to server (handles Insitu/Combo Eval)
    :param predictions: predictions outputted from classifier
    :param probs: probabilities of each image
    :return: n/a
    '''

    # Load dataframe
    if TARGET:
        file_name = '/data4/plankton_wi17/mpl/target_domain/aspect_target_image_path_labels.txt'
        df = pd.read_csv (file_name, sep=';', header=None)
        df.columns = ['path', 'img_id', 'gtruth']
    else:
        file_name = inputfile_dir + '/test.txt'
        df = pd.read_csv(file_name,sep=' ' ,header=None)
        df.columns = ['path','gtruth']

    # Add predictions to additional column
    df['predictions']= predictions

    # Calculate side lobes
    S = np.sort(probs)
    S = S[::-1]
    confidence_level = [(S[i,1]-S[i,0])/S[i,1] for i in range(len(S))]
    df['confidence_level'] = confidence_level

    # Save changes to csv output
    csv_filename = os.path.join(dest_path, CLASSIFIER_NAME + '-' + EXP_NUM + '_pred.csv')
    df.to_csv(csv_filename)

def write_pred2txt(predictions, probs, dest_path):
    '''
    Write predictions and confidence level to txt file to upload to server (used for Bench Eval)
    :param predictions: predictions outputted from classifier
    :param probs: probabilities of each image
    :return: n/a
    '''
    results_txt = open(dest_path  + '/Image_preds.txt', 'w')
    for i in range(len(predictions)):
        results_txt.write(str(probs[i,0])+' '+str(probs[i,1]))
        results_txt.write(' ')
        results_txt.write(str(predictions[i]))
        results_txt.write('\n')
    results_txt.close()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

if __name__=='__main__':

    # Choose which LMDB to run on
    if TARGET:
        test_data = 'aspect_target_fourhrs.LMDB'
    else:
        test_data = 'test1.LMDB'
    main (test_data, 2)
