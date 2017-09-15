Author: Kevin Le

Highlights of what experiments were done using this classifier and their respective results

#############################################################################
######################## Use Exp7 Model for best results ####################
#############################################################################

Classifier: bench_from_scratch
Exp 1: Hyperparameter Tuning
- base_lr: 0.001
Results: Best accuracy 89.1660727

Exp 2: Hyperparameter Tuning                (FAILED - ERROR WITH DATA MGT)
- base_lr: 0.01
Results: Classifier misclassified all actual positives as negatives
- Not trained properly?

Exp 3: Hyperparameter Tuning                (FAILED - ERROR WITH DATA MGT)
- base_lr: 0.0001
Results: Training accuracy/loss showed same results as Exp 2
- Model_exp3 not created
- Did not run on traget
- Ended experiment early around 2500 iteration

Exp 4: Hyperparameter Tuning                (FAILED - ERROR WITH DATA MGT)
- base_lr: 0.001
- Validating any possible error with training
Results: Training accuracy/loss showed same results as Exp 2
- Did not run on target
- Ended experiment early around 2500 iteration

Exp 5-6: Hyperparameter Tuning              (FAILED - ERROR WITH DATA MGT)
- Not sure what was changed
Results: Training accuracy/loss showed same results as Exp2
- Did not run on target


Exp7: Preserve Aspect Ratio and Resize
Results: Best accuracy 91.1528866714
