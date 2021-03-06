# SPCBench
Source Domain: SPCBench

# Introduction
The SPCBench experiments was set up to establish a baseline for the performance of classifiers trained on bench images, when comparing against the results of other source domains.
This baseline would be chosen out of two experiments, where one classifier is trained from scratch and the other fine-tuned to classifying the bench images. At the end of the project
the bench_finetune classifier offered the best performance and would be representative of the SPCBench classifiers.
This directory is meant for all of the files used in the experiments for the SPCBench source domain. It contains the following:

1. Classifiers
2. Python scripts to retrieve images from database and organize LMDB files for network input
3. Dataset statistics

# Classifiers
These are the classifiers that were used for this source domain.

1. bench_finetune
2. bench_from_scratch

Their respective directories contains the following:
a. Caffe prototxt files from each experiment
b. train, val, and test text files that contain the image files and labels
c. README.txt file to summarize each experiment

# How to use
Assuming you have access to the bench images through the server, simply run the plankclass_CopyImages.py script to download your dataset for these experiments.
Adjust path roots in create_LMDB-Bench.py to designated image directory, then run the script assuming the csv file is in the same folder.
Next would be to train the classifier using the train_alexnet.py script in the main folder and the caffenet file.
Finally, the evaluation script will be there to generate your results.

Please take these instructions with a grain of salt, so please be diligent with reviewing the scripts prior to running them.

# Experiment Results
Contact me @ kevin.le@gmail.com for access