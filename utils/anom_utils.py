import numpy as np
import torch.nn as nn
import sklearn.metrics as sk
import sklearn.neighbors
import sklearn.ensemble
import time
import torch
from torch.autograd import Variable
import os.path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


recall_level_default = 0.95

class ToLabel(object):
     def __call__(self, inputs):
        return (torch.from_numpy(np.array(inputs)).long())


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1] ## index from high to low
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))
    if np.array_equal(classes, [1]):
        return thresholds[cutoff]  # return threshold

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]


def get_measures(_pos, _neg, args, recall_level=recall_level_default, plot=True, save=True):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr, threshold = fpr_and_fdr_at_recall(labels, examples, recall_level)

    # Save results
    if save:
        np.save(f'{args.save_dir}/metrics', {'fpr': fpr, 'auroc': auroc, 'aupr': aupr})
        np.save(f'{args.save_dir}/in_scores', _pos)
        np.save(f'{args.save_dir}/out_scores', _neg)

    # Plot
    if plot:
        plot_kde(labels, examples, args)
        plot_roc(labels, examples, auroc, args)
        plot_pr(labels, examples, aupr, args)

    return auroc, aupr, fpr, threshold


def print_measures(auroc, aupr, fpr, args, recall_level=recall_level_default):
    print('\t\t\t' + args.ood+'_'+args.method)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr), flush=True)
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc), flush=True)
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr), flush=True)


def get_and_print_results(in_scores, out_scores, args, plot, save):
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(in_scores, out_scores, args, plot=plot, save=save)

    aurocs.append(measures[0])
    auprs.append(measures[1])
    fprs.append(measures[2])

    auroc = np.mean(aurocs)
    aupr = np.mean(auprs)
    fpr = np.mean(fprs)

    print_measures(auroc, aupr, fpr, args)
    return auroc, aupr, fpr, measures[3]


def get_localoutlierfactor_scores(val, test, out_scores):
    scorer = sklearn.neighbors.LocalOutlierFactor(novelty=True)
    print("fitting validation set")
    start = time.time()
    scorer.fit(val)
    end = time.time()
    print("fitting took ", end - start)
    val = np.asarray(val)
    test = np.asarray(test)
    out_scores = np.asarray(out_scores)
    print(val.shape, test.shape, out_scores.shape)
    return scorer.score_samples(np.vstack((test, out_scores)))


def get_isolationforest_scores(val, test, out_scores):
    scorer = sklearn.ensemble.IsolationForest()
    print("fitting validation set")
    start = time.time()
    scorer.fit(val)
    end = time.time()
    print("fitting took ", end - start)
    val = np.asarray(val)
    test = np.asarray(test)
    out_scores = np.asarray(out_scores)
    print(val.shape, test.shape, out_scores.shape)
    return scorer.score_samples(np.vstack((test, out_scores)))


def plot_roc(y_true, y_preds, auroc, args):
    fpr, tpr, _ = sk.roc_curve(y_true, y_preds)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % auroc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'{args.save_dir}/roc_curve.png')
    plt.close()


def plot_pr(y_true, y_preds, aupr, args):
    precision, recall, thresholds = sk.precision_recall_curve(y_true, y_preds)
    plt.title('Precision Recall Curve')
    plt.plot(recall, precision, 'b', label='AUC = %0.3f' % aupr)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(f'{args.save_dir}/pr_curve.png')
    plt.close()


def plot_kde(y_true, y_preds, args):
    df = pd.DataFrame(columns=['Score', 'Distribution'])
    df['Distribution'] = y_true
    df.loc[df['Distribution'] == 1, 'Distribution'] = 'In Distribution'
    df.loc[df['Distribution'] == 0, 'Distribution'] = 'Out-of-Distribution'
    df['Score'] = y_preds
    _, ax = plt.subplots()
    g = sns.displot(df, x="Score", hue="Distribution", kind="kde", fill=True, facet_kws=dict(legend_out=False))
    # ax.legend().set_title(None)
    sns.move_legend(g, "upper center", title=None)
    # g.legend().set_title(None)
    # plt.legend(loc='upper left')
    # sns.move_legend(g, "upper right")
    plt.savefig(f'{args.save_dir}/kde.png')
    plt.close()

def plot_histogram(y_true, y_preds, args):
    df = pd.DataFrame(columns=['Score', 'Distribution'])
    df['Distribution'] = y_true
    df.loc[df['Distribution'] == 1, 'Distribution'] = 'In Distribution'
    df.loc[df['Distribution'] == 0, 'Distribution'] = 'Out-of-Distribution'
    df['Score'] = y_preds
    _, ax = plt.subplots()
    g = sns.displot(df, x="Score", hue="Distribution", kind="kde", fill=True, facet_kws=dict(legend_out=False))
    # ax.legend().set_title(None)
    sns.move_legend(g, "upper center", title=None)
    # g.legend().set_title(None)
    # plt.legend(loc='upper left')
    # sns.move_legend(g, "upper right")
    plt.savefig(f'{args.save_dir}/kde.png')
    plt.close()