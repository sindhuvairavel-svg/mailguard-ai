# Model comparison (TEST set, evaluated once)
Test set did not influence model selection or thresholds.

| Model | Acc | Prec | Rec | F1 | FP | FN |
|---|---|---|---|---|---|---|
| MultinomialNB | 0.9733 | 1.0000 | 0.7403 | 0.8507 | 0 | 20 |
| ComplementNB | 0.9666 | 0.8514 | 0.8182 | 0.8344 | 11 | 14 |
| LogisticRegression | 0.9933 | 0.9615 | 0.9740 | 0.9677 | 3 | 2 |
| LinearSVM_calibrated | 0.9920 | 0.9733 | 0.9481 | 0.9605 | 2 | 4 |
| SGD | 0.9906 | 0.9730 | 0.9351 | 0.9536 | 2 | 5 |
| RandomForest | 0.9786 | 1.0000 | 0.7922 | 0.8841 | 0 | 16 |
| SoftVoting | 0.9920 | 0.9733 | 0.9481 | 0.9605 | 2 | 4 |

**Winner LinearSVM_calibrated** with tuned threshold 0.5114: {'accuracy': 0.9919786096256684, 'precision': 0.9733333333333334, 'recall': 0.948051948051948, 'f1': 0.9605263157894737, 'tn': 669, 'fp': 2, 'fn': 4, 'tp': 73}