import numpy as np
from sklearn.metrics import confusion_matrix


def classification_metrics(ground_truth, predicted_classes):
    binary = np.unique(ground_truth).shape[0] == 2
    if binary:
        tn, fp, fn, tp = confusion_matrix(ground_truth, predicted_classes).ravel()
        accuracy = (tn + tp) / (tn + fp + fn + tp)
    else:
        cm = confusion_matrix(ground_truth, predicted_classes)
        fp = cm.sum(axis=0) - np.diag(cm)
        fn = cm.sum(axis=1) - np.diag(cm)
        tp = np.diag(cm)
        tn = cm.sum() - (fp + fn + tp)
        accuracy = fp.sum() / cm.sum()
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    if binary:
        metric_infomation = (
            "[Accuracy] {0:.2%}({1}/{2}) ".format(
                accuracy,
                tn + tp,
                tn + fp + fn + tp,
            )
            + "[Specificity] {0:.2%}({1}/{2}) ".format(
                specificity,
                tn,
                tn + fp,
            )
            + "[Sensitivity] {0:.2%}({1}/{2}) ".format(
                sensitivity,
                tp,
                tp + fn,
            )
            + "[F1 Score] {:.4f} ".format(f1)
            + "[PPV] {0:.2%}({1}/{2}) ".format(ppv, tp, tp + fp)
            + "[NPV] {0:.2%}({1}/{2}) ".format(npv, tn, tn + fn)
            + "[Confusion Matrix] {0}".format(
                [tn, fp, fn, tp],
            )
        )
    else:
        metric_infomation = (
            "[Accuracy] {0:.2%}({1}/{2}) ".format(accuracy, tp.sum(), len(ground_truth))
            + "[Specificity] {0:.2%}({3}/{6}) - {1:.2%}({4}/{7}) - {2:.2%}({5}/{8}) ".format(
                *specificity, *tn, *(tn + fp)
            )
            + "[Sensitivity] {0:.2%}({3}/{6}) - {1:.2%}({4}/{7}) - {2:.2%}({5}/{8}) ".format(
                *sensitivity, *tp, *(tp + fn)
            )
            + "[F1 Score] {0:.4f} - {1:.4f} - {2:.4f} ".format(*f1)
            + "[PPV] {0:.2%}({3}/{6}) - {1:.2%}({4}/{7}) - {2:.2%}({5}/{8}) ".format(
                *ppv, *tp, *(tp + fp)
            )
            + "[NPV] {0:.2%}({3}/{6}) - {1:.2%}({4}/{7}) - {2:.2%}({5}/{8}) ".format(
                *npv, *tn, *(tn + fn)
            )
        )
    return (
        metric_infomation,
        accuracy,
        specificity,
        sensitivity,
        f1,
        ppv,
        npv,
        [tn, fp, fn, tp],
    )


"""
knee
fold_1 [1 1 1 0 1 0 1 1 1 0 0 1 0 0 0 1 1 1 1 1 0 0 0 1 1 0 0 0 0 1 0]
fold_2 [1 1 1 0 0 0 1 1 1 0 0 1 0 0 0 1 1 1 1 1 0 0 0 1 1 0 0 0 0 1 0]
fold_3 [1 1 1 0 0 1 1 1 1 0 0 1 0 0 1 1 1 1 1 1 0 0 0 1 1 0 0 0 0 1 0]
fold_4 [1 1 1 0 0 0 1 1 0 0 0 0 0 0 0 1 1 1 1 1 0 0 0 1 1 0 0 0 0 1 0]
fold_5 [1 1 1 0 0 0 1 1 1 0 0 1 0 0 1 1 1 1 1 1 0 0 0 1 1 0 1 0 0 1 0]
hip
fold_1 [0 0 0 0 0 0 0 0 1 0 1 0 1 0 0 1 0 0 1 1 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 0 0 0 1 1 0 0 0 0]
fold_2 [0 0 0 0 0 1 0 0 1 0 1 0 1 0 1 1 0 0 1 1 1 0 1 1 0 1 1 0 1 0 0 1 0 1 0 0 0 0 1 1 0 0 0 0]
fold_3 [0 0 0 0 0 1 0 0 0 0 1 0 1 0 0 1 0 0 1 1 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 0 0 0 1 1 0 0 0 0]
fold_4 [0 0 0 1 0 1 0 0 0 0 1 0 1 0 1 1 0 0 1 1 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 0 0 0 1 1 0 0 0 0]
fold_5 [0 0 0 0 1 1 0 0 1 0 1 0 1 0 1 1 0 1 1 1 1 0 0 1 0 1 1 0 1 0 0 1 0 1 0 0 0 0 1 1 0 0 0 0]
"""
