Author: Kevin Le

Highlights of what experiments were done using this classifier and their respective results

#############################################################################
######################## Use Exp4 Model for consistency #####################
#############################################################################

classifier: insitu_from_scratch
Exp1: Hyperparameter tuning
- base: 0.001
Evaluated test set at 80% accuracy

Exp2: Hyperparameter tuning
- base: 0.0001
Network converged around 60% learning accuracy -> Failed Exp (did not evaluate on test set)

Exp3: Hyperparamter tuning (FAILED - Error Data Mgt)
- Did not run on target

Exp4: Preserve Aspect Ratio
- Validated results at 84.6487424111
- Target normalized accuracy 82.5357
