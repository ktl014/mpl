# SPCombo
Source Domain: SPCombo

# Introduction
After establishing baselines for the other two source domains, the SPCombo experiments would focus on testing the validity of our hypothesis. To recall on the hypothesis, it is to exploit the use
of laboratory (bench) images when training classifiers, by alleviating the amount of in situ images to be labeled. See the technical report for more details.
This directory is meant for all of the files used in the experiments for the SPCombo source domain. It contains the following:

1. Combo Experiments
2. Python scripts to retrieve images from database and organize LMDB files for network input
3. Dataset statistics

# Combo Experiments
Unlike the other two source domains, this page is first organized by the combination experiments. Each combination experiment will contain a series of classifiers
trained on different partitions of the experimental source domain for the respective combination experiment.
These are the combination experiments that were used for this source domain.

1. all-noise100
2. bench-noise100
3. insitu-noise100

The classifier directories of the combination experiments contains the following:
a. Caffe prototxt files from each experiment
b. train, val, and test text files that contain the image files and labels
c. README.txt file to summarize each experiment

# How to use
Assuming you have the train, val, test txt files of your bench and insitu datasets, organize them into a source domain directory (i.e. spcbench, spcinsitu)
Next, simply run the create_dataset.py script to copy all of your images to a designated local directory.
>> python create_dataset.py
Adjust path roots in create_LMDB-Bench.py to designated image directory, then run the script and all of the datasets for each combination experiment will be generated.
Next would be to train the classifier using the train_alexnet.py script in the main folder and the caffenet file.
Finally, the evaluation script will be there to generate your results.

Please take these instructions with a grain of salt, so please be diligent with reviewing the scripts prior to running them.

# Experiment Results
Contact me @ kevin.le@gmail.com for access