import argparse
import datetime
import importlib.util
import json
import os
from pathlib import Path

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

TRACKER_PATH = (Path(__file__).resolve().parents[2] / "stg-stsg-model" / "src" / "experiment_tracker.py")


def _noop(*args, **kwargs):
    pass


if TRACKER_PATH.exists():
    spec = importlib.util.spec_from_file_location("stg_stsg_model_experiment_tracker", TRACKER_PATH)
    tracker_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tracker_module)
    log_results = tracker_module.log_results
    append_global_log = tracker_module.append_global_log
    plot_progress = tracker_module.plot_progress
else:
    log_results = append_global_log = plot_progress = _noop



def evaluate_dataset(cfg):
    results = []
    dataset_dir = cfg.get('dataset')
    if not dataset_dir:
        raise ValueError('Dataset path must be provided via config or --dataset.')
    files = [f for f in os.listdir(dataset_dir) if f.endswith("_pred.json")]

    for f_pred in tqdm(files, desc="Evaluating"):
        import re
        # normalize names like scene_000_depth_pred.json → scene_000_gt.json
        f_gt = re.sub(r"(_(depth|mask))?_pred\.json$", "_gt.json", f_pred)
        with open(os.path.join(dataset_dir, f_pred), "r", encoding="utf-8") as f:
            pred = json.load(f)
        f_gt_path = os.path.join(dataset_dir, f_gt)
        if not os.path.exists(f_gt_path):
            shared_gt = os.path.join('datasets/synthetic_facades', f_gt)
            if os.path.exists(shared_gt):
                f_gt_path = shared_gt
            else:
                print(f'[warn] missing GT for {f_pred}, skipping.')
                continue
        with open(f_gt_path, 'r', encoding='utf-8') as f:
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
    results_path = os.path.join(outdir, "results.csv")
    df.to_csv(results_path, index=False)

    # --- Auto experiment logging ---
    summary_path = os.path.join(outdir, "summary.png")
    outdir_abs = os.path.abspath(outdir)
    run_base = os.path.join(os.path.dirname(outdir_abs), 'experiments')
    os.makedirs(run_base, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    run_dir = os.path.join(run_base, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    config_guess = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "stg-stsg-model", "configs", "v1_facades.yaml"))
    log_results(run_dir, results_path, summary_path, config_path=config_guess)
    log_path = os.path.join(run_base, 'experiments_log.csv')
    append_global_log(log_path, timestamp, results_path)
    plot_progress(log_path)
    print(f'\n📊 Experiment logged: {run_dir}\n')

    print(df.describe())
    return df


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/synthetic_facades.yaml')
    parser.add_argument('--dataset', default='datasets/synthetic_facades',
                        help='dataset folder containing *_pred.json and *_gt.json')
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else {}
    if args.dataset:
        cfg['dataset'] = args.dataset
    evaluate_dataset(cfg)


if __name__ == "__main__":
    _main()
