from __future__ import print_function
import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.metrics import *
import caffe
import lmdb
import pandas as pd
import os
import timeit
import csv
import sys
import cPickle as pickle

# Specify which source domain & classifier will be used for evaluating the target domain
source = 'spcinsitu'
classifier = 'insitu_finetune'
exp_num = 'exp2'
model = 'model_' + exp_num + '.caffemodel'
domain_path = os.path.join ('/data4/plankton_wi17/mpl/source_domain/', source, classifier)
outroot = os.path.join('/data4/plankton_wi17/mpl/target_domain',source,classifier,exp_num)
# domain_path = lab_google_root


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

def eval_results():
    objects = []

    f1 = open('gtruth','r')
    gt = pickle.load(f1)
    f1.close()
    #print(gtruth.reshape((11224,1)))

    f2 = open('pred','r')
    pred = pickle.load(f2)
    f2.close()
    print(pred.shape)
    #print(pred.reshape((11224,1)))

    f3 = open('prob','r')
    prob = pickle.load(f3)
    f3.close()
    prob = np.concatenate(prob,0)
    print(prob.shape)

    accuracy = (pred == gt).mean()*100
    print(accuracy)

    #tp, tn, fn, fp = compute_cmatrix(gtruth,pred,2)

    tn, fp, fn, tp = confusion_matrix(gt,pred).ravel() #,labels=['copepod','non-copepod'])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 2
    pred_roc = np.zeros_like (prob)
    pred_roc[np.arange (len (prob)), prob.argmax (1)] = 1

    gtruth = np.zeros((len(gt),2))
    for i in range(len(gt)):
        if gt[i]==1:
            gtruth[i,1] = 1
        else:
            gtruth[i,0] = 1

    #fpr, tpr,_ = roc_curve (gtruth, pred, pos_label=0)
    for i in range(n_classes):
        fpr[i],tpr[i],_ = roc_curve(gtruth[:,i],prob[:,i])
        roc_auc[i] = auc (fpr[i], tpr[i])

    #fpr["micro"], tpr["micro"], _ = roc_curve (gtruth.ravel (), pred.ravel ())

    fpr["micro"], tpr["micro"], _ = roc_curve (gtruth.ravel (), prob.ravel ())
    roc_auc["micro"] = auc (fpr["micro"], tpr["micro"])

    #plot_roc_curve(n_classes,fpr,tpr)
    plot_roc_curve(n_classes,fpr,tpr,roc_auc)

def plot_roc_curve(gt,prob,classifier,num_class):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 2
    pred_roc = np.zeros_like (prob)
    pred_roc[np.arange (len (prob)), prob.argmax (1)] = 1

    gtruth = np.zeros((len(gt),2))
    for i in range(len(gt)):
        if gt[i]==1:
            gtruth[i,1] = 1
        else:
            gtruth[i,0] = 1

    #fpr, tpr,_ = roc_curve (gtruth, pred, pos_label=0)
    for i in range(num_class):
        fpr[i],tpr[i],_ = roc_curve(gtruth[:,i],prob[:,i])
        roc_auc[i] = auc (fpr[i], tpr[i])

    #fpr["micro"], tpr["micro"], _ = roc_curve (gtruth.ravel (), pred.ravel ())

    fpr["micro"], tpr["micro"], _ = roc_curve (gtruth.ravel (), prob.ravel ())
    roc_auc["micro"] = auc (fpr["micro"], tpr["micro"])

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
    plt.savefig(outroot + '/' + classifier+'_curve.png')
    print (classifier+'_curve.png saved.')

def compute_cmatrix(gtruth,pred,num_class):
    # Create array for confusion matrix with dimensions based on number of classes
    confusion_matrix = np.zeros((num_class,num_class))

    # Create confusion matrix
    for t,p in zip(gtruth,pred):
        confusion_matrix[t,p] += 1

    # Assign outcomes of confusion matrix
    true_positive = confusion_matrix[0,0]
    true_negative = confusion_matrix[1,1]
    false_positive = confusion_matrix[0,1]
    false_negative = confusion_matrix[1,0]

    print("True Positive\tFalse Negative")
    print(true_positive,'\t\t',false_negative)
    print ("False Positive\tTrue Negative")
    print (false_positive, '\t\t', true_negative)

    return true_positive,true_negative,false_negative,false_positive


def main(test_data, num_class, domain, classifier, model):

    gpu_id = 1

    t1 = timeit.default_timer() # Start timer

    # Load LMDB
    images, labels = load_lmdb(test_data)

    # Set to GPU mode
    caffe.set_mode_gpu()
    #caffe.set_device(gpu_id)

    # Create path to deploy protoxt and weights
    deploy_proto = os.path.join(domain, 'code/caffenet/deploy.prototxt')
    trained_weights = os.path.join(domain,'code',model)

    # Check if files can be found
    if not os.path.exists(deploy_proto):
        raise ValueError (os.path.basename(deploy_proto) + " not found")
    elif not os.path.exists(trained_weights):
        raise ValueError (os.path.basename(trained_weights) + " not found")
    elif not os.path.exists(outroot):
        os.makedirs(outroot)

    # Load net
    deploy = caffe.Net(deploy_proto,caffe.TEST, weights=trained_weights)
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

    # List to array
    probs = np.concatenate(probs, 0)

    print ('probs shape after concatenate:', probs.shape)
    print (probs[0,:], type(probs[0,0]))

    # compute accuracy
    predictions = probs.argmax (1)
    gtruth = np.array (labels)
    total_accu = (predictions == gtruth).mean () * 100

    print ('predictions shape:', predictions.shape)
    print (predictions[0:25])
    print ('Total Accuracy', total_accu)

    # # write prediction results to a txt file
    # pred_file_name = os.path.join(outroot,classifier+'_target_preds.txt')
    # results_txt = open(pred_file_name, 'w')
    # for i in range(len(gtruth)):
    #     results_txt.write(str(probs[i,0])+' '+str(probs[i,1]))
    #     results_txt.write(' ')
    #     results_txt.write(str(predictions[i]))
    #     results_txt.write('\n')
    # results_txt.close()

    # Write predictions to img path lbl csv file
    write_pred2csv(predictions,probs,classifier)

    # Plot ROC Curve
    plot_roc_curve(gtruth,probs,classifier,num_class)

    # Create array for confusion matrix with dimensions based on number of classes
    confusion_matrix_count = np.zeros ((num_class, num_class))
    class_count = np.zeros((num_class,1)) # 1st col represents number of images per class

    # Create confusion matrix
    for t, p in zip (gtruth, predictions):
        class_count[t,0] += 1
        confusion_matrix_count[t, p] += 1
    print(confusion_matrix_count)
    confusion_matrix_rate = np.zeros((num_class,num_class))
    for i in range(num_class):
        confusion_matrix_rate[i,:] = (confusion_matrix_count[i,:])/class_count[i,0]
    confusion_matrix_rate = np.around(confusion_matrix_rate, decimals=4)

    # Calculate Precision Rate
    precision = (confusion_matrix_count[0,0]/(confusion_matrix_count[0,0]+confusion_matrix_count[1,0]))*100 # TP / (FP+TP)
    print("Precision {}".format(precision))

    # Calculate side lobes
    S = np.sort (probs)
    S = S[::-1]
    confidence_level = [(S[i, 1] - S[i, 0]) / S[i, 1] for i in range (len (S))]
    confidence_level = np.asarray(confidence_level)
    avg_confidence = confidence_level.mean()*100
    print ("Confidence Level {}".format(avg_confidence))

    results_filename = os.path.join (outroot, classifier + '_Results.csv')
    outfile = open (results_filename, 'wb')
    writer = csv.writer (outfile, delimiter=",")
    writer.writerow (['Binary Classifier: ' + domain.split ('/')[6]])
    writer.writerow (['Total Accuracy'])
    writer.writerow ([str (total_accu)])
    writer.writerow (['Precision Rate'])
    writer.writerow ([str (precision)])
    writer.writerow (['Confidence Level'])
    writer.writerow ([str (avg_confidence)])
    writer.writerow (['Prediction Results:'])
    np.savetxt (outfile, confusion_matrix_count,'%5.2f', delimiter=",")
    writer.writerow (['Confusion Matrix:'])
    np.savetxt (outfile, confusion_matrix_rate * 100,'%5.2f', delimiter=",")
    print ('Print to', results_filename, 'file successful.')

def write_pred2csv(predictions, probs, classifier):
    '''
    Write predictions and confidence level to csv file to upload to server
    :param predictions: predictions outputted from classifier
    :param probs: probabilities of each image
    :return: n/a
    '''

    # Load dataframe
    file_name = 'master-fourhr_target_image_path_labels.csv'
    df1 = pd.read_csv(file_name,index_col=0)

    # Add predictions to additional column
    df1['predictions']= predictions

    # Calculate side lobes
    S = np.sort(probs)
    S = S[::-1]
    confidence_level = [(S[i,1]-S[i,0])/S[i,1] for i in range(len(S))]
    df1['confidence_level'] = confidence_level

    # Save changes to csv output
    df1.to_csv(outroot + '/' + classifier + '_preds.csv')

def temp():
    # Access text files for gtruth and pred
    with open (outroot + '/' + classifier +'_target_preds.txt', 'r') as f:
        df = pd.read_csv (f, sep=" ", header=None)
        df.columns = ['prob1', 'prob2', 'pred']
    f.close ()
    with open ('target_image_paths_labels.txt', 'r') as f_target:
        df1 = pd.read_csv (f_target, sep=" ", header=None)
        df1.columns = ['path', 'gtruth']
    f_target.close ()

    gtruth_list = df1['gtruth'].tolist()
    pred_list = df['pred'].tolist()
    path_list = df1['path'].tolist()

    fig = plt.figure(1,figsize=(15, 15))
    # True positive images [0,0]
    view_classified_img(fig, gtruth_list,pred_list,path_list, 0, 0, 1)
    #view_classified_img(gtruth_list,pred_list,path_list, 0, 0, 1,classifier+'_true_positive.png')

    # True negative images [1,1]
    view_classified_img (fig, gtruth_list, pred_list, path_list, 1, 1, 11)
    #view_classified_img (gtruth_list, pred_list, path_list, 1, 1, 2,lassifier+'_true_negative.png')

    # False positive images [1,0]
    view_classified_img (fig, gtruth_list, pred_list, path_list, 1, 0, 21)
    #iew_classified_img (gtruth_list, pred_list, path_list, 1, 0, 3, classifier + '_false_positive.png')

    # False negative images [0,1]
    view_classified_img (fig, gtruth_list, pred_list, path_list, 0, 1, 31)
    #view_classified_img (gtruth_list, pred_list, path_list, 0, 1, 4, classifier + '_false_negative.png')

    plt.savefig(outroot + '/' + classifier+'_img_summary', bbox_inches='tight')

def view_classified_img(fig,gtruth,pred,path,tf,pn,position):
    '''

     :param fig: figure to save all results on
     :param gtruth: ground truth labels
     :param pred: prediction results
     :param path: server path to image
     :param tf: true --> 0, false --> 1
     :param pn: positive --> 0. negative --> 1
     :param position: position to begin plotting images in row/col of figure
     :return: 
     '''
    n_row = 4  # number of outcomes on confusion matrix
    n_col = 10  # number of images per row
    end_row = position + 10
    for i in range (len (pred)):

        # Based off of true/false postive/negative, add image to subplot
        if gtruth[i] == tf and pred[i] == pn:
            img = mpimg.imread (path[i])
            fig.add_subplot (n_row, n_col, position)
            plt.imshow (img)
            plt.axis ('off')

            # Title: Index;Gtruth;Pred
            title = str (i + 1) + ';' + str (gtruth[i]) + ';' + str (pred[i])
            plt.title (title)
            position += 1

            # Stop collecting images after end of the row (10 columns)
            if position == end_row:
                break
if __name__=='__main__':
    main ('target_fourhrs.LMDB', 2, domain_path, classifier, model)
    #temp()