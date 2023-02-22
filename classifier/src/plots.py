import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from src.config import *
from sklearn.metrics import (PrecisionRecallDisplay, auc, roc_curve)


def plot_roc(ax, results):
    assert(len(results['y_holdout_list']) == len(results['y_proba_list']))
    num_folds = len(results['y_proba_list'])

    roc_tprs, roc_aucs = [], []
    for ix_fold in range(num_folds):
        y_hold_out = results['y_holdout_list'][ix_fold]
        y_proba = results['y_proba_list'][ix_fold]

        roc_fpr, roc_tpr, _ = roc_curve(y_hold_out, y_proba)
        roc_mean_fpr = np.linspace(0, 1, 100)
        roc_interp_tpr = np.interp(roc_mean_fpr, roc_fpr, roc_tpr)
        roc_interp_tpr[0] = 0.0

        roc_auc = auc(roc_fpr, roc_tpr)

        roc_tprs.append(roc_interp_tpr)
        roc_aucs.append(roc_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=1.1, color="black", label="Random Baseline", alpha=1)

    roc_mean_fpr = np.linspace(0, 1, 100)
    roc_mean_tpr = np.mean(roc_tprs, axis=0)
    roc_mean_tpr[-1] = 1.0
    roc_mean_auc = auc(roc_mean_fpr, roc_mean_tpr)
    roc_std_auc = np.std(roc_aucs)

    ax.plot(roc_mean_fpr, roc_mean_tpr, color="blue", label=r"Mean AUC=%0.2f$\pm$%0.2f" % (roc_mean_auc, roc_std_auc), alpha=1, lw=1)

    std_tpr = np.std(roc_tprs, axis=0)
    tprs_upper = np.minimum(roc_mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(roc_mean_tpr - std_tpr, 0)
    ax.fill_between(roc_mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=0.1, lw=1,
                    label=r'$\pm$ 1 SD (%i Folds)' % (num_folds))

    ax.legend(loc="lower right")
    ax.title.set_text('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')


def plot_sample_complexity(ax, train_cv_scores, test_cv_scores, train_sizes, n_folds):
    train_scores_mean = np.mean(train_cv_scores, axis=1)
    test_scores_mean = np.mean(test_cv_scores, axis=1)
    test_scores_std = np.std(test_cv_scores, axis=1)

    ax.plot(
        train_sizes, train_scores_mean, 'x--', color="black", alpha=1, lw=1,
        label="Training score")

    ax.plot(
        train_sizes, test_scores_mean, 'o-', color="blue", alpha=1, lw=1,
        label='CV Score (%i Folds)' % (n_folds))

    ax.fill_between(
        train_sizes, test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, color="blue", alpha=0.1, lw=1)

    ax.legend(loc="lower right")
    ax.title.set_text('Empirical Sample Complexity')
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("ROC AUC")


def plot_ap(results):
    assert(len(results['y_holdout_list']) == len(results['y_proba_list']))
    num_folds = len(results['y_proba_list'])

    _, ax = plt.subplots(1, 1, figsize=(5,4))
    for ix_fold in range(num_folds):
        y_hold_out = results['y_holdout_list'][ix_fold]
        y_proba = results['y_proba_list'][ix_fold]
        PrecisionRecallDisplay.from_predictions(y_hold_out, y_proba, name="Fold {}".format(ix_fold), alpha=1, lw=1, linestyle="-", ax=ax)

    ax.plot([0, 1], [0.5, 0.5], linestyle="--", lw=1.1, color="black", label="Baseline", alpha=0.99)
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    ax.legend(loc="lower left")

    plt.show()


def plot_drop_in_auc(ax, baseline_cv_scores, featuregroup_cv_scores, labels=False, sorted=False, err=True, rotation=0, ha='center'):
    baseline_mean = np.mean(baseline_cv_scores)
    losses = [baseline_mean - value for value in featuregroup_cv_scores.values()]
    loss_stds = [np.std(loss) for loss in losses]
    loss_means = [-np.mean(loss) for loss in losses]

    ix = np.arange(len(loss_means))
    if sorted:
        ix = np.argsort(loss_means)
    
    feature_labels = [key for key in featuregroup_cv_scores.keys()]

    if not labels:
        labels = np.array(feature_labels)[ix].tolist()
    labels = np.array(labels)[ix]

    values = np.array(loss_means)[ix].tolist()
    
    print(labels)
    print(np.round(values,3))

    errs = np.array(loss_stds)[ix].tolist()

    ax.axhline(0, color='blue', alpha=1, lw=1, linestyle='--', label='Baseline (%.2f)' % (baseline_mean))

    ax.bar(range(len(values)), values, yerr=errs if err else None, align='center', color='blue', alpha=0.16, lw=1, ecolor='black', capsize=5)

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: round(x + baseline_mean, 2)))
    
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=rotation, ha=ha)

