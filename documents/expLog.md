# SPC Bench/Insitu Experiments Revisited

Background: Spcbench 
![baseline.png](plots/baseline.png)

Normalized Error Rate (average of true positives errors to counter class imbalance)

* spcbench: 27.051%
* spcinsitu: 12.311%
* spcombo: 14.793%

## Goal: 
* Improve spcbench or spcombo performance to beat spcinsitu performance

## Procedure:
1) Add initially excluded Bench Copepod specimens to datasets and redo SPCBench & SPCombo experiments
2) Compare refined SPCBench images with original SPCBench images
3) Artifically augment the network to train for robustness against blurry images

### 1) Adding more data
#### SPCBench Dataset Statistics
Below are the new dataset statistics after adding labeled specimens that were collected and labeled July 2017. It was not initially included, because the labeling process was delayed.

Original Dataset -> New Dataset
Copepod
* 27 Copepod specimens -> 59 Copepod specimens ~ 2x more specimens
* 10464 Copepod images -> 23469 Copepod images

Non-Copepod
* 40 -> 60 Non-Copepod specimens
* 17513 Non-Copepod images -> 24072 Non-Copepod images

New ratio of Copepod to NonCopepod is approximately 1:1

Note: these are the training set sizes, that the models were trained on. 

| Calanoida        | Cyclopoida & Poecilomastoida          |
| ------------- |:-------------:|
| ![calanoida.png](plots/calanoidaAdditions.png)     | ![cyclopoida.png](plots/cyclopoidaAdditions.png)|
*Red boxes indicate new specimens

#### SPCInsitu Dataset Statistics
* 1,642 Copepod Images
* 2,382 Non-Copepod Images

Ratio of Copepod to Non-Copepod is approximately 1.4:1

100 random samples of each class below

| Copepod        | Non-Copepod           |
| ------------- |:-------------:|
| ![insituCopepod.png](plots/insituCopepod.png)      | ![insituNonCopepod.png](plots/insituNonCopepod.png) |

#### SPCombo Dataset Statistics
* 27957 Copepod Images (26308 spcbench, 1649 spcinsitu)
* 2382  Non-Copepod Images (0 spcbench, 2382 spcinsitu)

For visualization of this dataset, please refer to the images above for the SPCBench and SPCInsitu dataset.


#### Target Dataset Statistics
The target dataset is also pulled from the same source of images as the SPCInsitu, hence the similarity in images.

* 2,720 Copepod Images
* 23,049 Non-Copepod Images

Ratio of Copepod to Non-Copepods is 1:8

Below is a visualization of the target dataset.

| Copepod        | Non-Copepod           |
| ------------- |:-------------:|
| ![targetCopepod.png](plots/targetCopepod.png)      | ![targetNonCopepod.png](plots/targetNonCopepod.png) |

#### Dataset Statistics Summary
Below is a complete summary of all four datasets that are mentioned above. 
Again, these are all the training sizes of each domain, while the target statistics represents the testing size.

| SourceDomains | **Total Copepod** | Bench          | Insitu  | **Total  Non-Copepod** | Bench         | Insitu   |
| ------------- |:-------------:|:-------------: |:-------:|:-------------:|:-------------:|:-------:|
| SPCBench      | 23469         | 23469          |   ---   | 24072         | 24072       |   ---   |
| SPCInsitu     | 1649          | ---            |   ---   | 2382          | 0          |   2382   |
| SPCombo       | 27957         | 26308          | 1649    | 2382           | 0       |   2382   |
| Target        | 2720          | ---            | 2720    | 23049       | 0       |   23049   |
 

### Experiment Protocol
Below details steps taken into training and testing our model

#### Preprocessing
All datasets were normalized to a uniform size of 256x256, then resized using a perspective preserving transform along the major axis of the ROIs, and finally center-cropped
to train the classifier for robustness towards incomplete images.

#### Conv1 - Finetuning
This is an additional experiment suggested by Pedro to help boost the classifier's performance to different image properties, such as color balance, illumination, or poor resolution.
This transfer learning takes place by using the weights of our pretrained models on the Copepod and NonCopepods and freezing the layers, except Conv1, because this is presumably the layer
that takes into account these image properties. 

#### Evaluation Metrics
For our evaluation, we use a normalized error rate, that averages the true positive error rate of the copepod and non-copepod classes. This accounts for the class imbalance within the test set.

### Results
| SourceDomains        | Baseline           | New dataset           | Conv1 - Finetuning   |
| ------------- |:-------------:|:-------------:|:-------:|
| SPCBench      | 27.051%       | 24.824%       |   ---   |
| SPCInsitu     | 12.311%       | ---           |   ---   |
| SPCombo       | 13.052%       | 11.325%       | 11.457% |

Remarks from Confusion Matrix Analysis
* (SPCBench) 15% increase for true positive detection, but tradeoff with less precision, implying less confidence with predicting a copepod. 85% of the time, it will correctly classify a copepod.
* (SPCombo) Better with identifying true negatives by 7% and reached same precision as Insitu Model. Overall better model.
* All models have very similar recall rate of 85%, implying a representation problem within the test set, that is not included in the training set.

### 2) Optimizing SPCBench images

During discussions of results from our previous classification experiments, we decided to investigate a new direction with trying to improve image properties of the SPCBench system
to boost the utility of SPCBench images, when annotating SPCInsitu images. We will be referring to these refined images as BenchV1b and original as BenchV1a. Below
is a visualization of the V1a and V1b images.

| V1a      | V1b      | SPCInsitu    |
| :-----: | :-----: | :-----: |
| ![benchv1a.png](plots/benchv1aVisuals.png)      | ![benchv1b](plots/benchv1bVisuals.png)       | ![insitu.png](plots/spcInsitu.png)    |

Dataset Statistics
* Constrained to 7 specimens, which were randomly sampled for V1a and all specimens for V1b were all utilized.
* V1a: 2752 Copepod images, V1b: 2000 Copepod images
* Data combined with SPCInsitu Images (Copepod: 2356 images, NonCopepod: 3404 images)

Image Preprocessing
* No grayscale color conversion was utilized here to test the raw utility of the Bench images

### Results
|   | V1a  | V1b  |
|---|:---:|:---:|
| Normalized Error Rate  | 26.44%  | 19.10%  |
| Confusion Matrix  | ![v1aCM.png](plots/allv1c-noise100CM.png) | ![v1bCM](plots/allv1d-noise100CM.png) |

Analysis
* 15% increase in true positive detection for copepods, implying that similar image properties to SPCInsitu does play a role in classification performance
* Results succeeded in this area, but there is still a strong class imbalance in the test set. 

### Next Steps
* Fix test set
* Hard Negative Example Mining: 