# SPCInsitu
Source Domain: SPCInsitu

# Introduction
The SPCInsitu experiments was set up to establish a baseline for the performance of classifiers trained on in situ images, when comparing against the results of other source domains.
This baseline would be chosen out of two experiments, where one classifier is trained from scratch and the other fine-tuned to classifying the in situ images. At the end of the project
the insitu_finetune classifier offered the best performance and would be representative of the SPCInsitu classifiers.
This directory is meant for all of the files used in the experiments for the SPCInsitu source domain.


# Classifiers
These are the classifiers that were used for this source domain.

1. insitu_finetune
2. insitu_from_scratch

Their respective directories contains the following:
a. Caffe prototxt files from each experiment
b. train, val, and test text files that contain the image files and labels
c. README.txt file to summarize each experiment