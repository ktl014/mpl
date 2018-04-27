import glob, os, shutil
import pandas as pd
from collections import defaultdict
import numpy as np

"""
image database directory hierarchy:
images -> class** -> subclass** -> train/val/...
"""

IMAGE_SRC_PATH = '/data4/plankton_sp17/image_orig'
IMAGE_DEST_PATH = '/data4/plankton_wi17/mpl/source_domain/spcbench/bench_images'
#TIMESTAMP_LINK_DEST_PATH = '/data4/plankton_wi17/plankton/plankton_binary_classifiers/plankton_phytoplankton/image_softlinks/intervals/exp1'
DEBUG = False

def generate_softlinks(img_src_path):


    # Get list of images to be turned into softlinks
    src_path_list = sorted (glob.glob (os.path.join (img_src_path, '*')))

    # Uncomment time stamp function below and comment out orig src_path_list if only specific time stamp folders are needed
    '''
    # List of specimen time stamps
    time_stamp_list = ['20170217/001', '20170217/002', '20170130/001', '20170209/002', '20170223/002', '20170126/003',
              '20170203/001', '20170223/001', '20170207/001', '20170209/001', '20170207/002', '20170213/003',
              '20170214/002', '20170217/004', '20170224/010', '20170214/003', '20170214/005', '20170216/001',
              '20170224/004', '20170216/004', '20170217/006', '20170124/002', '20170213/001', '20170216/002',
              '20170217/003', '20170224/005', '20170224/012']
    src_path_list = [os.path.join(img_src_path,time_stamp) for time_stamp in time_stamp_list]
    '''

    # dir listing: src --> subdir_src --> image
    for src_path in src_path_list:
        subdir_src_list = sorted([os.path.join(src_path,subdir) for subdir in os.listdir(src_path)])
        for subdir_src_path in subdir_src_list:

            # Set up path to copy image-softlinks to
            subdir_src_path_parts = subdir_src_path.split('/')
            if DEBUG:
                print subdir_src_path_parts
            src_name = subdir_src_path_parts[4]
            img_subdir_name = subdir_src_path_parts[5]
            print('Accessing folder ' + src_name + ' ' + img_subdir_name)
            softlink_dest = os.path.join (IMAGE_DEST_PATH,src_name, img_subdir_name)
            if DEBUG:
                print softlink_dest
            img_dir_list = [os.path.join(subdir_src_path, fn) for fn in os.listdir(subdir_src_path)]


            # Make a folder for the image destination if not created
            if not os.path.exists (softlink_dest):
                os.makedirs (softlink_dest)
    
            # Generate softlink of each image
            for img_fn in img_dir_list:
                os.symlink(img_fn,os.path.join(softlink_dest,os.path.basename(img_fn)))


def create_dataset(img_src_path):
    #plank_category_list = ['good_proro','bad_proro']
    class_size = len(plank_category_list)
    img_src_path_list = [os.path.join(IMAGE_SRC_PATH,plank_category) for plank_category in plank_category_list]
    class_path_dest = range(len(img_src_path_list))
    class_label = range(len(img_src_path_list))
    for i, img_path in enumerate(img_src_path_list):
        if not os.path.exists(img_path):
            raise ValueError('Path does not exist ' + img_path)
        img_list = glob.glob(os.path.join(img_path,'*'))
        if i < 10:
            class_label[i] = '0' + str(i)
        else:
            class_label[i] = str(i)
        print 'class #: ', class_label[i]
        class_path_dest[i] = os.path.join(IMAGE_DEST_PATH, 'class'+class_label[i])
        if not os.path.exists(class_path_dest[i]):
            print 'Make directory: ', class_path_dest[i], '\n'
            os.makedirs(class_path_dest[i])
        print 'Copying images from ' + img_path
        #[shutil.copy(img_fn,class_path_dest) for img_fn in img_list]
        #for img_fn in img_list:
            #shutil.copy(img_fn,class_path_dest[i])
    print('Generating Labels')
    generate_labels(class_path_dest,class_label)

def generate_labels(class_path_dest_list,label_list):
    with open(os.path.join(os.path.dirname(IMAGE_DEST_PATH),'train.txt'),'w') as f:
        for i,class_path_dest in enumerate(class_path_dest_list):
            img_list = sorted(glob.glob(os.path.join(class_path_dest,'*')))
            for img in img_list:
                f.write(os.path.basename(img)+' '+label_list[i]+'\n')
    f.close()

def get_classes(txt_file):

    def is_copepod(row):
        copepod_classes = ['Calanoida', 'Cyclopoida']
        if row['order'] in copepod_classes:
            return 0
        else:
            return 1

    classes = pd.read_csv(txt_file)
    classes.loc[:, 'label'] = classes.apply (is_copepod, axis=1)

    return classes

def get_subsets(srcpath, specimen_list):
    subsets = defaultdict(list)
    for specimen in specimen_list:
        list_filename = os.path.join(srcpath, '%s-{}.lst' % specimen)
        for phase in ['train', 'valid', 'test']:
            for image in open(list_filename.format(phase)).read().splitlines():
                subsets['images'].append(unicode(image))
                subsets['phase'].append(phase)
                subsets['specimen_id'].append(specimen)

    return pd.DataFrame(subsets)

def cleanDataset(datasetDF):
    # Drop any unknown specimens
    cleanDataset = datasetDF[(datasetDF['order'] != 'UNKNOWN') &
                                                     (datasetDF['specimen_id'] != '20171207_006') &
                                                     (datasetDF['specimen_id'] != '20171130_003')]
    return cleanDataset

def createNewBench(datasetDF):
    isCopepod = (datasetDF['label'] == 0)
    isNonCopepod = (datasetDF['label'] == 1)
    specimenIDToFilter = '20171128'

    # Create new datasets
    newDatasetDF = datasetDF[(datasetDF['specimen_id'] >= specimenIDToFilter) & isCopepod]
    baselineDatasetDF = datasetDF[(datasetDF['specimen_id'] < specimenIDToFilter) & isCopepod]

    randomSmpl = np.random.choice(baselineDatasetDF['specimen_id'].unique(), size=7, replace=False)
    filteredBaseline = baselineDatasetDF[baselineDatasetDF['specimen_id'].isin(randomSmpl)]

    return filteredBaseline, newDatasetDF

def getDatasetSizes(datasetDF, phase):
    assert 'phase' in datasetDF.columns

    print('{} set size: {}, (Copepod:{}, NonCopepod:{})'.format(phase,
                                   len(datasetDF[datasetDF['phase'] == phase]),
          len(datasetDF[(datasetDF['phase'] == phase) & (datasetDF['label'] == 0)]),
              len(datasetDF[(datasetDF['phase'] == phase) & (datasetDF['label'] == 1)])
    ))


def main():
    # access list of images
    # print('Generating softlinks')
    # generate_softlinks(IMAGE_SRC_PATH)

    #print('Creating dataset')
    #create_dataset(IMAGE_SRC_PATH)
    newDatasetVersion = True

    root = '/data5/Plankton_wi18/rawcolor_db2'
    specimen_list = sorted(os.listdir(root + '/images'))
    classes = get_classes(os.path.join(root, 'classes/specimen_taxonomy.txt'))
    subsets = get_subsets(root + '/subsets', specimen_list)
    dataset = pd.merge(classes, subsets, on='specimen_id')

    dataset = cleanDataset(dataset)

    if newDatasetVersion:
        baseline, newDataset = createNewBench(dataset)

    datasetName, datasets = {'baseline': baseline, 'newData': newDataset}, {}
    for i in datasetName:
        print('Dataset: {}'.format(i))
        for phase in ['train', 'valid', 'test']:
            dataset = datasetName[i]
            datasets['phase'] = dataset[dataset['phase'] == phase]
            getDatasetSizes(dataset, phase)
            destDatasetPath = '/data4/plankton_wi17/mpl/source_domain/spcbench/bench_data/V1c/{}'.format(i)
            datasetFileName = destDatasetPath + '/{}.csv'.format(phase)
            #datasets['phase'].to_csv(datasetFileName)
            print('{} dataset saved. Find at {}\n'.format(phase, datasetFileName))

class PlanktonDataset(object):
    def __init__(self, csv_filename, img_dir, phase):
        self.data = pd.read_csv(csv_filename)
        self.img_dir = img_dir
        self.phase = phase

    def get_fns(self):
        shuffle_images = (self.phase == 'train' or self.phase == 'val')
        if shuffle_images:
            self.data = self.data.iloc[np.random.permutation(len(self.data))]
            self.data = self.data.reset_index(drop=True)

        self.fns = list(self.img_dir + self.data['images'])
        self.lbls = np.array(self.data['label'])
        return self.fns, self.lbls

if __name__ == "__main__":
    main()