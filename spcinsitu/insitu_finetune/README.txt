Author: Kevin Le

Highlights of what experiments were done using this classifier and their respective results

#############################################################################
######################## Use Exp8 Model for consistency #####################
#############################################################################

Classifier: insitu_finetune
Exp 1: Hyperparameter Tuning
- base_lr: 0.001
Results:
Validation Accuracy 87.25
Target Accuracy 92.1184368815
Precision Rate 60.3298350825

Exp 2: Hyperparameter Tuning
- base_lr: 0.0001
Results:
Validation Accuracy 88.8117953166
Target Accuracy 91.8196282355
Precision Rate 58.2302313072
Confidence Level 82.9343676567

Exp 3: Finetune with spcbench weights and freeze conv1-5 lr. Parameters same as exp2
spcbench_model: '/data4/plankton_wi17/mpl/source_domain/spcbench/bench_finetune/code/model_exp2.caffemodel'
Results:
Validation Accuracy 87.8577623591
Target Accuracy 90.3449881641
Precision Rate 52.8445316332

Exp4: Combine insitu training set with target set as training set for insitu_finetune
Validation Accuracy 85.6027753686
Target Accuracy 95.78951453
Precision Rate 81.38195777
Confidence Level 92.39860177

Exp5: Hyperparameter Tuning
- LR 0.0001
- Max Iter: 10000
- Stepsize: 2500
- Learning rate decay: 0.1

Exp6: Hyperparameter Tuning
- LR 0.0001
- Max Iter: 10000
- Stepsize: 2500
- Learning rate decay: 0.5

Exp7: Hyperparameter Tuning
- LR 0.0001
- Max Iter: 10000
- Stepsize: 2500
- Learning rate decay: 0.75
Validation Accuracy 86.81699913
Target Accuracy 91.86619582

Exp8: Preserved aspect ratio while resizing the image (see img.jpg and img_og.jpg as examples)
- Used same hyperparameter as experiment 6

