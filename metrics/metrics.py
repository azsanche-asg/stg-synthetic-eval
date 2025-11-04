import numpy as np


def rule_f1(pred, gt):
    tp = len(set(pred["rules"]) & set(gt["rules"]))
    fp = len(set(pred["rules"]) - set(gt["rules"]))
    fn = len(set(gt["rules"]) - set(pred["rules"]))
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    return 2 * prec * rec / (prec + rec + 1e-8)


def regularity_error(pred, gt):
    return np.mean(np.abs(np.array(pred["repeats"]) - np.array(gt["repeats"])))


def mdl_score(pred):
    return len(pred["rules"]) + 0.5 * pred["depth"]


def persistence_score(pred, gt):
    return np.mean([i in gt["persist_ids"] for i in pred["persist_ids"]])


def motion_error(pred, gt):
    if "motion" not in pred or "motion" not in gt:
        return np.nan
    return np.mean(np.abs(np.array(pred["motion"]) - np.array(gt["motion"])))
