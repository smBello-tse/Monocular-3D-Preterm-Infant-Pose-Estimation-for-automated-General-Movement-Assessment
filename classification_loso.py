import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from sklearn.metrics import auc, roc_curve, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut, KFold

from classification import best_f1_threshold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="m3d/evals", help="Path to m3d results")
    parser.add_argument("--file", type=str, default="m3d.csv", help="Name of m3d result file")
    parser.add_argument("--print", action="store_true", help="Print the results.")
    opts = parser.parse_args()
    return opts

class Patient:
    def __init__(self, name, scores, labels):
        self.name = name
        self.scores = scores if isinstance(scores, list) else [scores]
        self.labels = labels if isinstance(labels, list) else [labels]
        self.preds = []

    def add_prediction(self, pred):
        self.preds.append(pred)

    def __str__(self):
        return f"Patient: {self.name}\nLabels: {self.labels}\nPredictions: {self.preds}"


class PatientList:
    def __init__(self):
        self.patients = []

    def add_patient(self, patient):
        for existing_patient in self.patients:
            if existing_patient.name == patient.name:
                existing_patient.scores.extend(patient.scores)
                existing_patient.labels.extend(patient.labels)
                return

        self.patients.append(patient)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, i):
        return self.patients[i]

    def print_patients(self):
        if not self.patients:
            print("No patients in the list.")
            return

        for patient in self.patients:
            print(patient)
            print("-" * 30)

    def get_metrics(self):
        all_labels = []
        all_preds = []

        patient_metrics = {}

        for patient in self.patients:
            y_true = np.array(patient.labels)
            y_pred = np.array(patient.preds)

            if len(y_true) != len(y_pred):
                raise ValueError(
                    f"Mismatch for patient {patient.name}: "
                    f"{len(y_true)} labels vs {len(y_pred)} predictions"
                )

            # Store predictions for global metrics
            all_labels.extend(y_true)
            all_preds.extend(y_pred)

            # Per-patient confusion matrix
            tn, fp, fn, tp = confusion_matrix(
                y_true, y_pred, labels=[0, 1]
            ).ravel()

            patient_metrics[patient.name] = {
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "Accuracy": (tp + tn) / (tp + tn + fp + fn),
                "Sensitivity": tp / (tp + fn) if tp + fn > 0 else np.nan,
                "Specificity": tn / (tn + fp) if tn + fp > 0 else np.nan,
                "F1": f1_score(y_true, y_pred, zero_division=0)
            }

        # Global metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        if len(y_true) == 0:
            raise ValueError("No predictions available.")

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        global_metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Sensitivity": tp / (tp + fn) if tp + fn > 0 else np.nan,
            "Specificity": tn / (tn + fp) if tn + fp > 0 else np.nan,
            "F1": f1_score(y_true, y_pred, zero_division=0),
        }

        return global_metrics, patient_metrics

def loso(N):
    '''
    Return indices for leave one subject out.
    '''
    groups = np.arange(N)
    logo = LeaveOneGroupOut()

    X = np.zeros((N, 1))

    return list(logo.split(X, groups=groups)) #[(train0, test0), ..., (trainN, testN)]

def kfolds(N, K=5, shuffle=False, random_state=None):
    '''
    Return indices for K-fold cross-validation.
    '''
    kf = KFold(n_splits=K, shuffle=shuffle, random_state=random_state)

    X = np.zeros((N, 1))

    return list(kf.split(X))




###################################################################################################
if __name__ == "__main__":
    opts = parse_args()
    pred_path = opts.dir
    pred_file = opts.file

    # Get infants used for fine-tuning
    fine_tuning_infants = []
    subsets = ["train", "validation", "test"]
    for subset in subsets:
        ps = os.listdir(os.path.join("data/agma_3dhpe_annotated", subset))
        ps = [p.split("_")[0] for p in ps]
        ps = [".".join(p.split(".")[:-1]) for p in ps]
        fine_tuning_infants += ps
    fine_tuning_infants = list(set(fine_tuning_infants))
    fine_tuning_infants.sort()

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
    final_patient_lists = []


    assert len(gt_patient) == len(pred_patient)

    # We first need to get rid of the videos that were not assessable.
    for i in range(len(gt_patient)):
        assert gt_patient[i] == pred_patient[i]
        name = ".".join(gt_patient[i].split(".")[:-1])
        # if name in fine_tuning_infants:
        #     #print("skipping infant: ", name)
        #     continue
        if gt_decisions[i] == "1":
            good_idx.append(i)
            normal_idx.append(i)
            final_patient_lists.append(name)
        elif gt_decisions[i] == "0":
            abnormal_idx.append(i)
            good_idx.append(i)
            final_patient_lists.append(name)
    print("There are: ", len(good_idx), "videos")
    gt_m3d_normal = gt_m3d[normal_idx]
    preds_m3d_normal = preds_m3d[normal_idx]
    gt_m3d_abnormal = gt_m3d[abnormal_idx]
    preds_m3d_abnormal = preds_m3d[abnormal_idx]
    gt_decisions = gt_decisions[good_idx].astype(int)
    gt_m3d = gt_m3d[good_idx]
    preds_m3d = preds_m3d[good_idx]
    patients = PatientList()
    for name, score, label in zip(final_patient_lists, gt_decisions, preds_m3d):
        patients.add_patient(Patient(name, 1 - label, 1 - score)) # We want normal to be for low scores and also to be the class 0

    #patients.print_patients()
    N = len(patients)
    splits = loso(N)

    test_folds, best_thresholds, acc, f1, sensitivity, specificity, auc_, preds, gts = [], [], [], [], [], [], [], [], []

    for train_idx, test_idx in splits:
        # training
        scores = []
        labels = []
        for i in train_idx:
            scores += patients.patients[i].scores
            labels += patients.patients[i].labels
        scores = np.array(scores)
        labels = np.array(labels)
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        auc_.append(auc(fpr, tpr))
        best_threshold, _ = best_f1_threshold(fpr, tpr, thresholds, sum(labels),
                                              len(labels) - sum(labels))
        #print("Best threshold: ", 1 - best_threshold)

        # testing
        scores = np.array(patients.patients[test_idx[0]].scores)
        gts += np.array(patients.patients[test_idx[0]].labels).astype(int).tolist()
        patients.patients[test_idx[0]].preds = (scores > best_threshold).astype(int).tolist()

    global_metrics, patient_metrics = patients.get_metrics()

    print("\nPER PATIENT METRICS (ONLY FOR FINE-TUNING PATIENTS).")
    tps = 0
    fps = 0
    tns = 0
    fns = 0
    # for name, metrics in patient_metrics.items():
    #     if name in fine_tuning_infants:
    #         #print(name, metrics)
    #         tps += metrics["TP"]
    #         fps += metrics["FP"]
    #         tns += metrics["TN"]
    #         fns += metrics["FN"]
    # acc = (tps + tns) / (tps + tns + fns + fps)
    # sens = (tps) / (tps + fns)
    # spec = (tns) / (tns + fps)
    # f1 = (2 * tps) / (2 * tps + fps + fns)
    print(opts.dir)
    # print(f"Average accuracy: {np.mean(acc):.2f}, average sensitivity: {sens:.2f},  average specificity: {spec:.2f}; average F1: {np.mean(f1):.2f}")

    print("GLOBAL METRICS")
    for k, v in global_metrics.items():
        print(f"{k}: {v:.3f}")
    auc_ = np.array(auc_)
    #print(f"auc: {np.mean(auc_)} +/- {np.std(auc_, ddof=1) / len(auc_)**0.5}")
    scores = []
    labels = []
    for i in range(len(patients.patients)):
        scores += patients.patients[i].scores
        labels += patients.patients[i].labels
    scores = np.array(scores)
    labels = np.array(labels)
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    roc_data = {"fpr": fpr, "tpr": tpr}
    os.makedirs("rocs", exist_ok=True)
    method = opts.dir.split("/")[1]
    #np.savez(f"rocs/{method}.npz", **roc_data)
    # print(f"Diffs: accuracy: {acc - global_metrics['Accuracy']}, sensitivity: {sens - global_metrics['Sensitivity']}, specificity: {spec - global_metrics['Specificity']}, F1: {f1 - global_metrics['F1']}")
    print("############################################################")

