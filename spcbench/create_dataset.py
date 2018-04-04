import glob, os, shutil

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

def main():
    # access list of images
    # print('Generating softlinks')
    generate_softlinks(IMAGE_SRC_PATH)

    #print('Creating dataset')
    #create_dataset(IMAGE_SRC_PATH)

if __name__ == "__main__":
    main()