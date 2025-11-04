import argparse
import json
import os

import pandas as pd
import yaml
from tqdm import tqdm

from metrics.metrics import (
    mdl_score,
    motion_error,
    persistence_score,
    regularity_error,
    rule_f1,
)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_dataset(cfg):
    results = []
    dataset_dir = cfg["dataset"]
    files = [f for f in os.listdir(dataset_dir) if f.endswith("_pred.json")]

    for f_pred in tqdm(files, desc="Evaluating"):
        f_gt = f_pred.replace("_pred.json", "_gt.json")
        with open(os.path.join(dataset_dir, f_pred), "r", encoding="utf-8") as f:
            pred = json.load(f)
        with open(os.path.join(dataset_dir, f_gt), "r", encoding="utf-8") as f:
            gt = json.load(f)

        res = {
            "scene": f_pred.replace("_pred.json", ""),
            "rule_f1": rule_f1(pred, gt),
            "reg_error": regularity_error(pred, gt),
            "mdl": mdl_score(pred),
            "persist": persistence_score(pred, gt),
            "motion_err": motion_error(pred, gt),
        }
        results.append(res)

    df = pd.DataFrame(results)
    outdir = cfg["output_dir"]
    os.makedirs(outdir, exist_ok=True)
    df.to_csv(os.path.join(outdir, "results.csv"), index=False)
    print(df.describe())
    return df


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    evaluate_dataset(cfg)


if __name__ == "__main__":
    _main()
