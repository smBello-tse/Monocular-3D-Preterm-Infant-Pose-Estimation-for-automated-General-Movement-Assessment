import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from sklearn.metrics import auc
from scipy.stats import t

def corrected_kfold_ttest(v1, v2):
    """
    Corrected k-fold CV t-test (Nadeau-Bengio /
    k-fold-specific correction).
    """
    differences = np.asarray(v1) - np.asarray(v2)
    k = len(differences)

    n = len(differences)
    mean_diff = np.mean(differences)
    variance = np.var(differences, ddof=1)

    rho = 1 / k
    correction = (1 / n) + (rho / (1 - rho))

    t_stat = mean_diff / np.sqrt(correction * variance)

    p_value = 2 * t.sf(abs(t_stat), df=n - 1)

    return mean_diff, t_stat, p_value

methods = os.listdir("rocs")
ref_name = "pml_mod"
ref = np.load("rocs/triangulation.npz") if ref_name == "tri" else np.load("rocs/pml_mod_protocol_2_m3d_.npz")

plt.figure(figsize=(9, 7))
print("ref_name: ", ref_name)
for method in methods:

    if ref_name in method:
        continue

    print(method)
    target = np.load(f"rocs/{method}")

    for k in target.keys():
        if k not in ["Accuracy", "Sensitivity", "Specificity", "F1-score", "AUC"]:
            continue

        v1 = np.array(ref[k])
        v2 = np.array(target[k])
        mean_diff, t_stat, p_value = corrected_kfold_ttest(v1, v2)

        print(
            f"{k} fold: mean difference={mean_diff:.3f} "
            f"(p={p_value:.3f})"
        )
    print("#########################################")
