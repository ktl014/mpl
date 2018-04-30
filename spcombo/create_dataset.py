"""Description
Copy bench and insitu images to local dir and separate into their respective classes
Author: Kevin Le
"""
import os
import glob
import pandas as pd
import shutil
DEBUG = False

def create_dataset(datasetVersion):
    '''
    Separate bench and insitu images to their respective classes to make dataset for combo experiments
    :return: 
    '''
    datasets = ['spcbench','spcinsitu']

    # Loop through each source domain dir
    for dataset in datasets:
        print ("dataset {}".format(dataset))
        keys = ['train','val','test']

        # Loop through each <key>.txt file in
        for key in keys:
            print("starting {}".format(key))

            # Initialize path to read text file from
            src_path = os.path.join (os.getcwd (), dataset)

            # Read text file into dataframe to collect image paths and labels
            if dataset == 'spcinsitu':
                df = pd.read_csv(src_path + "/{}.txt".format(key), sep = " ", header=None)
                df.columns = ['path','label']
            elif dataset == 'spcbench':
                img_dir = '/data5/Plankton_wi18/rawcolor_db2/images/'
                df = pd.read_csv(src_path+ "/{}.csv".format(key))
                df['path'] = img_dir + df['images']
                df = df.drop('images', axis=1)

            # Initialize dir to copy images to
            img_root = '/data4/plankton_wi17/mpl/source_domain/spcombo/combo_images_exp/{}'.format(datasetVersion)
            num_class = 2
            for classes in range(num_class):

                # Obtain list of images for each class
                class_list = df['path'][df['label']==classes].tolist()
                print("number of files class {}: {}".format(classes,len(class_list)))

                # Initialize path to direct images to
                dest_path = os.path.join(img_root,dataset,'class{}'.format(classes))
                if not os.path.exists(dest_path):
                   os.makedirs(dest_path)

                # Copy all images in image list to desired dir
                f = open(dest_path + '/data{}.txt'.format(classes), 'w')
                for img in class_list:
                    f.write(img + '\n')
                    # img_basename = os.path.basename(img)
                    #shutil.copy(img,os.path.join(dest_path,img_basename)) # line to copy files
                f.close()

def separate_insitu_bench():
    img_root = '/data4/plankton_wi17/mpl/source_domain/spcombo/combo_images_exp'
    with open("stats.txt","w") as f:
        class_list = glob.glob(os.path.join(img_root,'*'))
        for classes in class_list:
            class_number = os.path.basename(classes)
            img_list = glob.glob(os.path.join(classes,'*'))
            for img in img_list:
                if img.startswith("SPC2"):
                    insitu_destpath = os.path.join(img_root,'spcinsitu',class_number)
                    if not os.path.exists(insitu_destpath):
                        os.makedirs(insitu_destpath)
                    shutil.move(img,insitu_destpath + '/{}'.format(img))
                elif img.startswith("SPCBENCH"):
                    bench_destpath = os.path.join(img_root,'spcbench',class_number)
                    if not os.path.exists(bench_destpath):
                        os.makedirs(bench_destpath)
                    shutil.move(img,bench_destpath + '/{}'.format(img))

    f.close()

def countimg():
    '''
    Loop through class dir to obtain dataset statistics
    :return: N/A
    '''
    img_root = '/data4/plankton_wi17/mpl/source_domain/spcombo/combo_images_exp'
    datasets = ['spcbench','spcinsitu']
    with open("stats.txt","w") as f:
        datasets = glob.glob(img_root + '/*')
        for dataset in datasets:
            print ("dataset {}".format(dataset))
            classlist = glob.glob(os.path.join( dataset,'*'))
            for classes in classlist:
                class_number = os.path.basename(classes)
                img_list = glob.glob(os.path.join(classes,'*'))
                f.write(dataset + '\n')
                f.write(class_number + " " + str(len(img_list)) + '\n')
    f.close()

if __name__ == '__main__':
    create_dataset(datasetVersion='V1b')

