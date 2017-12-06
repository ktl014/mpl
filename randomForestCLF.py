import matplotlib
matplotlib.use('Agg')
# Load scikit's random forest classifier library
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
# Load pandas
import pandas as pd

# Load numpy
import numpy as np

# Load feature extraction module
from spc_extractFeatures import *

# Load glob, os, random
import glob
import os
import random
import caffe
import pickle
DEBUG = False
import matplotlib.pyplot as plt
import itertools
import joblib
import csv
import timeit


# Grab list of each image
# Shuffle list of images
# Partition Images for 80%
# Write txt file of image and label
# Grab imgs and labels for test set
# Return Dict of images and labels

# Joblib save image feature matrix for training and testing sets
def main():
    t1 = timeit.default_timer()
    EXP_NUM = 'exp2'
    idx = 1
    DOMAIN = ['spcbench',
              'spcinsitu',
              'spcombo']
    CLASSIFIER_NAME = '{}_ensemble'.format(DOMAIN[idx].strip('spc'))
    SRCPATH = ['/data4/plankton_wi17/mpl/source_domain/spcombo/combo_images_exp',
               '/data4/plankton_wi17/mpl/source_domain/spcinsitu/insitu_finetune/code',
               '/data4/plankton_wi17/mpl/source_domain/spcombo/combo_finetune/all-noise100/all-noise100_100-100/code']
    train, test = train_testSplit(DOMAIN[idx], SRCPATH[idx])

    featurename = '{}tFeatures'.format (DOMAIN[idx])
    featuresExtracted = True
    if not featuresExtracted:
        # Extract features for training and testing set
        trainFeatures = extractFeature(train['imgs'])
        joblib.dump(trainFeatures, featurename)
        t2 = timeit.default_timer()
        print "{} finished extracting after {}".format(featurename, t2-t1)
        testFeatures = extractFeature(test['imgs'])
    else:
        trainFeatures = joblib.load (featurename)
        testFeatures = joblib.load('targetsetFeatures')

    print 'Number of training examples: {}'.format(len(trainFeatures))
    print 'Number of test examples: {}'.format(len(testFeatures))
    unique, counts = np.unique(train['lbls'], return_counts=True)
    print 'Examples per class: '
    print np.asarray((unique, counts))

    alpha_estimator = [10, 100, 250, 500, 1000, 2000, 5000, 7500]
    alpha_depth = [2, 5, 10, 20, 50, 70]
    alpha_minsample = [2, 5, 10, 20, 50, 70]
    alpha_kfolds = [5, 10, 15, 20, 50, 70]
    jobs = [2, 5, 7, 10, 15, 20]
    normError = []
    clf = RandomForestClassifier (n_estimators=alpha_estimator[3], max_depth=alpha_depth[3], min_samples_split=alpha_minsample[4], n_jobs=jobs[0], random_state=1)
    results = []
    for i in range(len(alpha_kfolds)):
        # Simple K-Fold cross validation
        cv = cross_validation.KFold(len(trainFeatures), n_folds=alpha_kfolds[i])
        dest_path = '/data4/plankton_wi17/mpl/target_domain/{}/{}'.format(DOMAIN[idx], CLASSIFIER_NAME) + '/{}'.format(EXP_NUM)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        num_classes = 2
        for traincv, testcv in cv:
            clf.fit(trainFeatures[traincv], train['lbls'][traincv])
            preds = clf.predict(testFeatures[testcv])
            err = evaluatePredictions (preds, test['lbls'][testcv], num_classes, dest_path, CLASSIFIER_NAME, EXP_NUM)
            normError.append(err)
        results.append(np.array(normError).mean())
        print "results: " + str(np.array(normError).mean())
    plt.figure()
    plt.plot(alpha_kfolds, results, 'b-'); plt.title('Probability of Error vs Number of KFolds'); plt.xlabel('Number of KFolds'); plt.ylabel('Probability of Error')
    plt.savefig('POE_kfolds.png')
    # for i in range(len(alpha_estimator)):
    #     clf = RandomForestClassifier (n_estimators=alpha_estimator[i], max_depth=alpha_depth[3], min_samples_split=alpha_minsample[4], n_jobs=jobs[0], random_state=1)
    #     # Train the Classifier to take the training features and learn how they relate to the training y (the species)
    #     clf.fit (trainFeatures, train['lbls'])
    #
    #     # Apply the Classifier we trained to the test data (which, remember, it has never seen before)
    #     preds = clf.predict (testFeatures)
    #     dest_path = '/data4/plankton_wi17/mpl/target_domain/{}/{}'.format(DOMAIN[idx], CLASSIFIER_NAME) + '/{}/{}'.format(EXP_NUM, EXP_NUM + '_%02d'%i)
    #     if not os.path.exists(dest_path):
    #         os.makedirs(dest_path)
    #     num_classes = 2
    #     err = evaluatePredictions(preds, test['lbls'], num_classes, dest_path, CLASSIFIER_NAME, EXP_NUM)
    #     normError.append(err)
    # plt.figure()
    # alpha = 'Estimator'
    # plt.plot(alpha_estimator, normError, 'b-'); plt.title('Probability of Error vs Number of {}'.format(alpha)); plt.xlabel('Number of {}'.format(alpha)); plt.ylabel('Probability of Error')
    # plt.savefig('PoE_{}.png'.format(alpha))



def evaluatePredictions(preds, gtruth, num_class, dest_path, CLASSIFIER_NAME, EXP_NUM):
    totalAccu = (preds == gtruth).mean()*100
    errorRate = (preds != gtruth).mean() * 100
    # print 'Total Accuracy:', totalAccu
    # print 'Error Rate:', errorRate

    # Create array for confusion matrix with dimensions based on number of classes
    confusion_matrix_rawcount = np.zeros ((num_class, num_class))
    class_count = np.zeros((num_class,1)) # 1st col represents number of images per class

    # Create confusion matrix
    predClass = ['Predicted Copepods', 'Predicted Non-Copepods']
    trueClass = ['True Copepods', 'True Non-Copepods']
    for t, p in zip (gtruth, preds):
        class_count[t,0] += 1
        confusion_matrix_rawcount[t, p] += 1
    df_rawcount = pd.DataFrame(confusion_matrix_rawcount, columns=predClass, index=trueClass)
    confusion_matrix_rate = np.zeros((num_class,num_class))
    for i in range(num_class):
        confusion_matrix_rate[i,:] = (confusion_matrix_rawcount[i,:])/class_count[i,0]*100
    confusion_matrix_rate = np.around(confusion_matrix_rate, decimals=4)
    df_rate = pd.DataFrame(confusion_matrix_rate, columns=predClass, index=trueClass)
    print df_rate

    # Normalized Accuracy
    normAccu = (confusion_matrix_rate[0,0]+confusion_matrix_rate[1,1])/2.00
    print("Normalized Accuracy: {}".format(normAccu))

    # Normalized Error Rate
    normError = 100.00 - normAccu
    print("Normalized Error: {}".format(normError))

    # Calculate Precision Rate
    precision = (confusion_matrix_rawcount[0,0]/(confusion_matrix_rawcount[0,0]+confusion_matrix_rawcount[1,0]))*100 # TP / (FP+TP)
    # print("Precision: {}".format(precision))

    # Calculate Recall Rate
    recall = (confusion_matrix_rawcount[0,0]/(confusion_matrix_rawcount[0,0]+confusion_matrix_rawcount[0,1]))*100 # TP / (FN+TP)
    # print("Recall: {}".format(recall))

    results_filename = os.path.join (dest_path, CLASSIFIER_NAME + '-' + EXP_NUM + '_Results.csv')
    outfile = open (results_filename, 'wb')
    writer = csv.writer (outfile, delimiter=",")
    writer.writerow (['Binary Classifier: {}-{}'.format(CLASSIFIER_NAME, EXP_NUM)])
    writer.writerow (['Total Accuracy:'])
    writer.writerow ([str (totalAccu)])
    writer.writerow (['Error Rate:'])
    writer.writerow ([str (errorRate)])
    writer.writerow (['Normalized Accuracy:'])
    writer.writerow ([str (normAccu)])
    writer.writerow (['Normalized Error Rate:'])
    writer.writerow ([str (normError)])
    writer.writerow (['Precision Rate:'])
    writer.writerow ([str (precision)])
    writer.writerow (['Recall Rate:'])
    writer.writerow ([str (recall)])
    writer.writerow (['Confusion Matrix (Raw Count):'])
    writer.writerow ([str(df_rawcount)])
    writer.writerow (['Confusion Matrix (Rate):'])
    writer.writerow ([str(df_rate)])
    print 'Print to', results_filename, 'file successful.'

    # Plot Confusion Matrix
    classNames = ['copepod', 'non-copepod']
    plt.figure()
    plt.subplot(211)
    plot_confusion_matrix(confusion_matrix_rawcount,classes=classNames,title='Confusion Matrix (Raw Count)')
    plt.subplot(212)
    plot_confusion_matrix(confusion_matrix_rate,classes=classNames,title='Confusion Matrix (Rate)')
    cnf_plot_filename = os.path.join(dest_path,'cnf_matrix.png')
    plt.savefig(cnf_plot_filename)

    return normError

def train_testSplit(domain, srcPath):
    assert isinstance(domain, str) # options limited to spcbench, spcinsitu, spcombo
    trainPartition = 0.70
    train_fns = []; train_lbls = []
    print domain + ' selected'
    if domain == 'spcbench':
        destPath = '/data4/plankton_wi17/mpl/source_domain/spcbench/bench_ensemble'
        srcpath = srcPath + '/{}'.format(domain)
        classList = glob.glob(srcpath + '/*')
        with open("{}_stats.txt".format(domain), "w") as f:
            for cl in classList:
                imgList = glob.glob(cl + '/*')
                cl_lbl = int(os.path.basename(cl).replace('class', ''))
                lblList = [cl_lbl]*len(imgList)

                # Shuffle retrieved image list before partitioning into train and test sets
                totalImgs = len(imgList)
                image_index = range (totalImgs)
                random.shuffle (image_index)
                imgList = [imgList[i] for i in image_index]
                f.write( 'class ' + str(cl_lbl) + ' ' + str(totalImgs) + '\n')
                trainSize = int(totalImgs * trainPartition)
                f.write('\t train: {} \n'.format(trainSize))
                train_fns += imgList[:int(totalImgs * trainPartition)]; train_lbls += [cl_lbl]*trainSize
        f.close()
    elif domain == 'spcinsitu':
        destPath = '/data4/plankton_wi17/mpl/source_domain/spcinsitu/insitu_ensemble'
        train_fns, train_lbls = getFnsAndLbls(srcPath + '/train.txt', Target=False)
    elif domain == 'spcombo':
        destPath = '/data4/plankton_wi17/mpl/source_domain/spcombo/combo_ensemble'
        train_fns, train_lbls = getFnsAndLbls(srcPath + '/train.txt', Target=False)

    testFile = '/data4/plankton_wi17/mpl/target_domain/aspect_target_image_path_labels.txt'
    test_fns, test_lbls = getFnsAndLbls(testFile, Target=True)

    train_fns, train_lbls = randomize_writepaths(train_fns, train_lbls, 'train', destPath)
    test_fns, test_lbls = randomize_writepaths(test_fns, test_lbls, 'test', destPath)
    train = {'imgs': train_fns, 'lbls': train_lbls}
    test = {'imgs': test_fns, 'lbls': test_lbls}
    return train, test

def getFnsAndLbls(textFile, Target=True):
    try:
        if Target:
            df = pd.read_csv(textFile, delimiter=';', names=['img','id', 'lbl'])
        else:
            df = pd.read_csv(textFile, delimiter=' ', names=['img', 'lbl'])
    except:
        print "Check text file formatting"
    test_fns = df['img'].tolist()
    test_lbls = np.array(df['lbl'].tolist())
    return test_fns, test_lbls

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

def extractFeature(imgList):
    numFeatures = 72
    featureVector = np.zeros([len(imgList), numFeatures])
    for i, imgname in enumerate(imgList):
        img = caffe.io.load_image (imgname)
        img = (img * 255).astype (np.uint8)
        featureVector[i,:] = extractFeatures(img)
        if i%100 == 0:
            print i,'/',len(imgList)
    return featureVector

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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

def datasetFeatureExtract():
    bench_srcpath = '/data4/plankton_wi17/mpl/source_domain/spcombo/combo_images_exp'  # Image dir for bench and insitu images
    train, test = train_testSplit('spcbench', bench_srcpath)
    testFeatures = extractFeature(test['imgs'])
    joblib.dump(testFeatures, 'benchsetFeatures')

def loadFeatures(filename):
    feature = joblib.load(filename)
    print type(feature)
    print feature.shape

if __name__ == '__main__':
    # main()
    main()

    # testsetFn = '/data4/plankton_wi17/mpl/source_domain/ensembleClassifier/targetsetFeatures'
    # loadFeatures(testsetFn)
    # train_testSplit(domain="spcbench")
    # getFnsAndLbls()



