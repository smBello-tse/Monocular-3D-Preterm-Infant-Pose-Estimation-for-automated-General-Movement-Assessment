import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import auc, roc_curve, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt

def plot_roc(fpr_gt, tpr_gt, fpr_preds, tpr_preds):
    """
    Plots ROC curves for ground truth and predicted scores.

    Parameters:
    - fpr_gt: False Positive Rates for ground truth.
    - tpr_gt: True Positive Rates for ground truth.
    - fpr_preds: False Positive Rates for predictions.
    - tpr_preds: True Positive Rates for predictions.
    """
    auc_gt = auc(fpr_gt, tpr_gt)
    auc_preds = auc(fpr_preds, tpr_preds)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_gt, tpr_gt, label=f'Ground Truth ROC (AUC = {auc_gt:.2f})', linestyle='--', color='blue')
    plt.plot(fpr_preds, tpr_preds, label=f'Prediction ROC (AUC = {auc_preds:.2f})', linestyle='-', color='green')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def youden_best_threshold(fpr, tpr, thresholds):
    '''
    Returns the best threshold, i.e. the one that maximises Youden index.

    Parameters:
        - fpr: False Positive Rates.
        - tpr: True Positive Rates.
        - thresholds: List of thresholds from ROC curve.
    '''

    youden_index = tpr - fpr
    best_threshold_index = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_index]

    #print(f"Best Threshold: {best_threshold}")
    return best_threshold, youden_index

def best_f1_threshold(fpr, tpr, thresholds, P, N):
    '''
    Returns threshold that maximizes F1 score.

    Parameters:
        - fpr: False Positive Rates.
        - tpr: True Positive Rates.
        - thresholds: List of thresholds from ROC curve.
        - P: Number of positive samples.
        - N: Number of negative samples.
    '''


    tp = tpr * P
    fn = P - tp
    fp = fpr * N
    f1 = 2 * tp / (2 * tp + fn + fp)
    best_idx = np.argmax(f1)
    return thresholds[best_idx], f1[best_idx]

def classic_ci(values, confidence=0.95):
    """
    Calculate confidence interval for a set of values using an analytic formula.
    
    Parameters:
        - values: Array of metric values from cross-validation folds
        - confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
    """
    alpha = 1 - confidence
    q = norm.ppf(1 - alpha/2, loc=0., scale=1.)
    mu = np.mean(np.array(values))
    sigma = np.std(np.array(values), ddof=1)  # For unbiased estimator.
    N = len(values)
    ci_lower = mu - q * sigma / (N ** 0.5)
    ci_upper = mu + q * sigma / (N ** 0.5)
    
    return ci_lower, ci_upper

def bootstrap_ci(values, n_bootstrap=1000, confidence=0.95, seed=42):
    """
    Calculate bootstrap confidence interval for a set of values.
    
    Parameters:
        - values: Array of metric values from cross-validation folds
        - n_bootstrap: Number of bootstrap samples
        - confidence: Confidence level (default 0.95 for 95% CI)
        - seed: Random seed for reproducibility
    
    Returns:
        - ci_lower: Lower bound of confidence interval
        - ci_upper: Upper bound of confidence interval
    """
    np.random.seed(seed)
    values = np.array(values)
    n = len(values)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    alpha = 1 - confidence
    ci_lower = np.percentile(bootstrap_means, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return ci_lower, ci_upper

def k_folds(normal_n, abnormal_n, k=5, seed=42):
    """
    Returns k folds of (train_indices, val_indices). To know which index belongs to which class, note that abnormal indexes start from normal_n and normal indexes end at normal_n - 1.

    Returns:
        folds: List of tuples [(train_idx, val_idx)]
    """
    np.random.seed(seed)

    # Assign class labels
    normal_idx = np.random.permutation(normal_n)
    abnormal_idx = np.random.permutation(abnormal_n) + normal_n  # offset to avoid overlap

    # Split into k folds
    normal_folds = np.array_split(normal_idx, k)
    abnormal_folds = np.array_split(abnormal_idx, k)

    folds = []

    for i in range(k):
        val_idx = np.concatenate((normal_folds[i], abnormal_folds[i]))
        train_idx = np.concatenate([
            np.concatenate(normal_folds[:i] + normal_folds[i + 1:]),
            np.concatenate(abnormal_folds[:i] + abnormal_folds[i + 1:])
        ])
        folds.append((train_idx, val_idx))

    return folds

def cross_validation(folds, scores_normal, scores_abnormal, n_bootstrap=1000):
    label_normal, label_abnormal = 0, 1
    true_labels = np.array([label_normal for i in range(scores_normal.shape[0])] + [label_abnormal for i in range(scores_abnormal.shape[0])])
    scores = np.concatenate([scores_normal, scores_abnormal])
    #print("True_labels: ", true_labels)
    #print("scores: ", scores)
    test_folds, best_thresholds, acc, f1, sensitivity, specificity, auc_ = [], [], [], [], [], [], []
    for i, (train, val) in enumerate(folds):
        test_folds.append(f"Fold {i + 1}")
        #print(f"Fold {i + 1}: Train={len(train)}, Val={len(val)}")

        #Training to determine the best threshold using auc
        scores_train = 1 - scores[train] #So that normal cases(0) correspond to score <= threshold.
        labels_train = true_labels[train]
        fpr, tpr, thresholds = roc_curve(labels_train, scores_train, pos_label=1)
        #print(thresholds)
        auc_.append(auc(fpr, tpr))
        #best_threshold, _ = youden_best_threshold(fpr, tpr, thresholds)
        best_threshold, _ = best_f1_threshold(fpr, tpr, thresholds, sum(labels_train), len(labels_train) - sum(labels_train))
        #best_threshold = 0.28

        #Testing threshold
        scores_val = 1 - scores[val]#So that normal cases(0) correspond to score <= threshold.
        labels_val = true_labels[val]
        pred_labels = (scores_val > best_threshold).astype(int)
        if i < 0:
            print(f"Pred_labels: {pred_labels}")
        #print("y_pred unique:", np.unique(pred_labels))
        #print("y_true unique:", np.unique(labels_val))
        cm = confusion_matrix(labels_val, pred_labels, labels=[0, 1])
        best_thresholds.append(1 - best_threshold)#The M3D threshold is 1 - roc_threshold, according to our prior transformation
        acc.append(accuracy_score(labels_val, pred_labels))
        f1.append(f1_score(labels_val, pred_labels, zero_division=0))
        specificity.append(cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm[0, 0] + cm[0, 1] > 0 else 0)
        sensitivity.append(cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm[1, 1] + cm[1, 0] else 0)

    # Calculate confidence intervals for each metric (both methods)
    ci_threshold_classic = classic_ci(best_thresholds)
    ci_threshold_bootstrap = bootstrap_ci(best_thresholds, n_bootstrap=n_bootstrap)
    
    ci_acc_classic = classic_ci(acc)
    ci_acc_bootstrap = bootstrap_ci(acc, n_bootstrap=n_bootstrap)
    
    ci_auc_classic = classic_ci(auc_)
    ci_auc_bootstrap = bootstrap_ci(auc_, n_bootstrap=n_bootstrap)
    
    ci_f1_classic = classic_ci(f1)
    ci_f1_bootstrap = bootstrap_ci(f1, n_bootstrap=n_bootstrap)
    
    ci_sensitivity_classic = classic_ci(sensitivity)
    ci_sensitivity_bootstrap = bootstrap_ci(sensitivity, n_bootstrap=n_bootstrap)
    
    ci_specificity_classic = classic_ci(specificity)
    ci_specificity_bootstrap = bootstrap_ci(specificity, n_bootstrap=n_bootstrap)

    #Add mean, std, and CI.
    test_folds.append("Mean")
    test_folds.append("Std")
    test_folds.append("CI Classic Lower")
    test_folds.append("CI Classic Upper")
    test_folds.append("CI Bootstrap Lower")
    test_folds.append("CI Bootstrap Upper")
    
    best_thresholds.append(np.array(best_thresholds).mean())
    best_thresholds.append(np.array(best_thresholds)[:-1].std())
    best_thresholds.append(ci_threshold_classic[0])
    best_thresholds.append(ci_threshold_classic[1])
    best_thresholds.append(ci_threshold_bootstrap[0])
    best_thresholds.append(ci_threshold_bootstrap[1])
    
    acc.append(np.array(acc).mean())
    acc.append(np.array(acc)[:-1].std())
    acc.append(ci_acc_classic[0])
    acc.append(ci_acc_classic[1])
    acc.append(ci_acc_bootstrap[0])
    acc.append(ci_acc_bootstrap[1])
    
    auc_.append(np.array(auc_).mean())
    auc_.append(np.array(auc_)[:-1].std())
    auc_.append(ci_auc_classic[0])
    auc_.append(ci_auc_classic[1])
    auc_.append(ci_auc_bootstrap[0])
    auc_.append(ci_auc_bootstrap[1])
    
    f1.append(np.array(f1).mean())
    f1.append(np.array(f1)[:-1].std())
    f1.append(ci_f1_classic[0])
    f1.append(ci_f1_classic[1])
    f1.append(ci_f1_bootstrap[0])
    f1.append(ci_f1_bootstrap[1])
    
    sensitivity.append(np.array(sensitivity).mean())
    sensitivity.append(np.array(sensitivity)[:-1].std())
    sensitivity.append(ci_sensitivity_classic[0])
    sensitivity.append(ci_sensitivity_classic[1])
    sensitivity.append(ci_sensitivity_bootstrap[0])
    sensitivity.append(ci_sensitivity_bootstrap[1])
    
    specificity.append(np.array(specificity).mean())
    specificity.append(np.array(specificity)[:-1].std())
    specificity.append(ci_specificity_classic[0])
    specificity.append(ci_specificity_classic[1])
    specificity.append(ci_specificity_bootstrap[0])
    specificity.append(ci_specificity_bootstrap[1])

    return {"Test_fold": test_folds, "M3D threshold": best_thresholds, "Accuracy": acc, "Sensitivity": sensitivity, "Specificity": specificity, "F1 score": f1, "AUC": auc_}

def leave_one_out(normal_n, abnormal_n, seed=42):
    """
    Returns folds for Leave-One-Sample-Out cross-validation.

    The index mapping:
        - Normal indices: 0 to normal_n - 1
        - Abnormal indices: normal_n to normal_n + abnormal_n - 1

    Returns:
        folds: List of tuples [(train_idx, val_idx)]
    """
    np.random.seed(seed)

    normal_idx = np.random.permutation(normal_n)
    abnormal_idx = np.random.permutation(abnormal_n) + normal_n

    all_idx = np.concatenate([normal_idx, abnormal_idx])
    folds = []

    for i in range(len(all_idx)):
        val_idx = np.array([all_idx[i]])
        train_idx = np.delete(all_idx, i)
        folds.append((train_idx, val_idx))

    return folds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="m3d/evals", help="Path to m3d results")
    parser.add_argument("--file", type=str, default="m3d.csv", help="Name of m3d result file")
    opts = parser.parse_args()
    return opts




###################################################################################################
opts = parse_args()
pred_path = opts.dir
pred_file = opts.file

# method3d = input("Method 3d? ")
augmentation = ""  # input("Augmentation? ")
df_gt = pd.read_excel("Decisions.xlsx")
df_preds = pd.read_csv(os.path.join(pred_path, pred_file))
df_gt = df_gt.sort_values("Video")
df_preds = df_preds.sort_values("Video")
# print(df_gt.head())
# print(df_preds.head())
gt_patient = df_gt["Video"].tolist()
pred_patient = df_preds["Video"].tolist()
gt_decisions = np.array(df_gt["Decision"].tolist())
gt_m3d = np.array(df_gt["M3D"].tolist())
preds_m3d = np.array(df_preds["all_joints"].tolist())
good_idx = []
normal_idx = []
abnormal_idx = []

assert len(gt_patient) == len(pred_patient)

# We first need to get rid of the videos that were not assessable.
for i in range(len(gt_patient)):
    assert gt_patient[i] == pred_patient[i]
    if gt_decisions[i] == "1":
        good_idx.append(i)
        normal_idx.append(i)
    elif gt_decisions[i] == "0":
        abnormal_idx.append(i)
        good_idx.append(i)
gt_m3d_normal = gt_m3d[normal_idx]
preds_m3d_normal = preds_m3d[normal_idx]
gt_m3d_abnormal = gt_m3d[abnormal_idx]
preds_m3d_abnormal = preds_m3d[abnormal_idx]
gt_decisions = gt_decisions[good_idx].astype(int)
gt_m3d = gt_m3d[good_idx]
preds_m3d = preds_m3d[good_idx]
print(f"There are {len(normal_idx)} normal cases and {len(abnormal_idx)} abnormal cases.")

# Cross-validation: k-folds
print("\n\n\n################################################\nK-folds cross-validation")
k = 5
folds = k_folds(len(normal_idx), len(abnormal_idx), k)
triang_results = pd.DataFrame(cross_validation(folds, gt_m3d_normal, gt_m3d_abnormal)).T

# Configure pandas display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

print("\n" + "="*80)
print("Triangulation results:")
print("="*80)
print(triang_results.to_string())
print("="*80 + "\n")

preds_results = pd.DataFrame(cross_validation(folds, preds_m3d_normal, preds_m3d_abnormal)).T
print("\n" + "="*80)
print(f"Results:")
print("="*80)
print(preds_results.to_string())
print("="*80 + "\n")
preds_results.to_excel(os.path.join(pred_path, "classification_results.xlsx"))