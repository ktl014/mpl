# Target
Target Domain

# Introduction
The Target domain is our test set for indepenently evaluating the performance of the source domain classifiers. The data is pulled from a different domain to avoid bias when evaluating the classifiers.
This directory is meant for all of the files used in the experiments for the SPCombo source domain. It contains the following:

1. Python scripts to retrieve images from database and organize LMDB files for network input
2. Dataset statistics

# How to use
Adjust pull_unlabeled_spcdata3.py script to designate where to download images and simply run the following command:

>> python pull_unlabeled_spcdata3.py time_period.txt < destination directory >

Adjust path roots in create_LMDB-Target.py to designated image directory, then run the script assuming the csv file is in the same folder.
This will generate your dataset for evaluation purposes.

Please take these instructions with a grain of salt, so please be diligent with reviewing the scripts prior to running them.
