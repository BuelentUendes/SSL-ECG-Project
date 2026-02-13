import argparse
import os
import json
import numpy as np
import pandas as pd

from utils.helper_paths import RESULTS_PATH

# Strategy names map CLI arg -> actual directory name
STRATEGY_TO_DIR = {
    "lp_ft": "fine_tuned_encoder_new_head",
    "head_only": "pretrained_encoder_new_head",
    "zero_shot": "zero_shot_performance",
}

# ---------------------------------------------------------------------------
# Path builders
# ---------------------------------------------------------------------------

def _result_file(base_path, strategy_dir, model, seed, label_fraction, window_size, step_size):
    """Return path to test_results.json for a transfer-learning model."""
    if model.startswith("cnn_"):
        head = model.split("_", 1)[1]
        return os.path.join(
            base_path, strategy_dir, "cnn",
            str(seed), str(label_fraction),
            str(window_size), str(step_size), head,
            "test_results.json",
        )
    else:
        return os.path.join(
            base_path, strategy_dir, model,
            "logistic_regression", str(seed), str(label_fraction),
            str(window_size), str(step_size),
            "test_results.json",
        )


def _lp_ft_file(base_path, strategy_dir, model, seed, window_size, step_size):
    """Return path to test_results_lp_ft.json."""
    if model.startswith("cnn_"):
        head = model.split("_", 1)[1]
        return os.path.join(
            base_path, strategy_dir, "cnn",
            str(seed), "1.0",
            str(window_size), str(step_size), head,
            "test_results_lp_ft.json",
        )
    else:
        return os.path.join(
            base_path, strategy_dir, model,
            "logistic_regression", str(seed), "1.0",
            str(window_size), str(step_size),
            "test_results_lp_ft.json",
        )


def _zero_shot_file(base_path, strategy_dir, model, seed, label_fraction):
    """Return path to zero_shot_results.json.

    Zero-shot CNN results are stored without a head subdirectory:
      cnn/{seed}/{label_fraction}/zero_shot_results.json
    """
    if model.startswith("feature_engineered_ws"):
        parts = model.split("_")
        ws = int(parts[2][2:])
        ss = int(parts[3][2:])
        return os.path.join(
            base_path, strategy_dir, "feature_engineered",
            "logistic_regression", str(seed), str(label_fraction),
            str(ws), str(ss),
            "zero_shot_results.json",
        )
    elif model == "cnn" or model.startswith("cnn_"):
        # Zero-shot CNN has no classifier-head subdir
        return os.path.join(
            base_path, strategy_dir, "cnn",
            str(seed), str(label_fraction),
            "zero_shot_results.json",
        )
    else:
        return os.path.join(
            base_path, strategy_dir, model,
            "logistic_regression", str(seed), str(label_fraction),
            "zero_shot_results.json",
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_transfer_metrics(file_path):
    with open(file_path) as f:
        data = json.load(f)
    m = data["test_metrics"]
    return {"auroc": m["auroc"], "pr_auc": m["pr_auc"]}


def _load_zero_shot_metrics(file_path):
    with open(file_path) as f:
        data = json.load(f)
    result = {
        "auroc": data["zero_shot_roc_auc"],
        "pr_auc": data["zero_shot_pr_auc"],
        "accuracy": data.get("zero_shot_accuracy"),
    }
    if "zero_shot_balanced_accuracy" in data:
        result["balanced_accuracy"] = data["zero_shot_balanced_accuracy"]
    if "zero_shot_f1_score" in data:
        result["f1_score"] = data["zero_shot_f1_score"]
    return result


def load_results(base_path, strategy_dir, model, seeds,
                 label_fraction=1.0, window_size=10, step_size=5,
                 is_zero_shot=False, use_lp_ft_file=False):
    """Load per-seed results for one model variant.

    use_lp_ft_file=True  -> reads test_results_lp_ft.json  (lp_ft strategy)
    is_zero_shot=True    -> reads zero_shot_results.json
    otherwise            -> reads test_results.json          (head_only strategy)
    """
    results = []
    for seed in seeds:
        if is_zero_shot:
            file_path = _zero_shot_file(base_path, strategy_dir, model, seed, label_fraction)
            loader = _load_zero_shot_metrics
        elif use_lp_ft_file:
            file_path = _lp_ft_file(base_path, strategy_dir, model, seed, window_size, step_size)
            loader = _load_transfer_metrics
        else:
            file_path = _result_file(base_path, strategy_dir, model, seed, label_fraction, window_size, step_size)
            loader = _load_transfer_metrics

        if os.path.exists(file_path):
            try:
                results.append({"seed": seed, **loader(file_path)})
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [WARN] Error reading {file_path}: {e}")
        else:
            print(f"  [MISSING] {file_path}")

    return results


def load_feature_results(base_path, dataset_name, window_size, step_size, seeds, label_fraction=1.0):
    """Load feature-engineered (logistic regression) results."""
    features_path = os.path.join(RESULTS_PATH, f"{dataset_name}_features")
    results = []
    for seed in seeds:
        file_path = os.path.join(
            features_path, str(window_size), str(step_size),
            "logistic_regression", str(seed), str(label_fraction),
            "test_results.json",
        )
        if os.path.exists(file_path):
            try:
                results.append({"seed": seed, **_load_transfer_metrics(file_path)})
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  [WARN] Error reading {file_path}: {e}")
        else:
            print(f"  [MISSING] {file_path}")
    return results


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _stats(values):
    n = len(values)
    mean = np.mean(values)
    std = np.std(values)
    return {"mean": mean, "std": std, "se": std / np.sqrt(n), "n": n}


def _print_metric(label, values):
    n = len(values)
    mean, std = np.mean(values), np.std(values)
    print(f"  {label}: {mean:.3f} ± {std:.3f}  (stderr: {std/np.sqrt(n):.4f})")


def _summary_row(model, metric, values):
    n = len(values)
    mean, std = np.mean(values), np.std(values)
    return {
        "model": model, "metric": metric,
        "mean": mean, "std": std,
        "stderr": std / np.sqrt(n),
        "n_seeds": n,
    }


# ---------------------------------------------------------------------------
# Result processor
# ---------------------------------------------------------------------------

def process_model_results(results, model_name, all_results, summary_stats, aggregated_results, is_zero_shot):
    """Accumulate stats for one model variant."""
    if not results:
        return

    auroc = [r["auroc"] for r in results]
    pr_auc = [r["pr_auc"] for r in results]

    print(f"\n{model_name}:")
    _print_metric("AUROC", auroc)
    _print_metric("PR-AUC", pr_auc)

    aggregated_results[model_name] = {
        ("zero_shot_roc_auc" if is_zero_shot else "auroc"): _stats(auroc),
        ("zero_shot_pr_auc" if is_zero_shot else "pr_auc"): _stats(pr_auc),
    }

    summary_stats.extend([
        _summary_row(model_name, "AUROC", auroc),
        _summary_row(model_name, "PR_AUC", pr_auc),
    ])

    if is_zero_shot:
        for metric_key, label in [
            ("accuracy", "Accuracy"),
            ("balanced_accuracy", "Balanced Accuracy"),
            ("f1_score", "F1 Score"),
        ]:
            values = [r[metric_key] for r in results if r.get(metric_key) is not None]
            if values:
                _print_metric(label, values)
                aggregated_results[model_name][f"zero_shot_{metric_key}"] = _stats(values)
                summary_stats.append(_summary_row(model_name, label.replace(" ", "_"), values))

    print(f"  Seeds: {[r['seed'] for r in results]}  (n={len(results)})")

    for r in results:
        all_results.append({"model": model_name, **r})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(base_path, strategy_dir, models, is_zero_shot=False,
         include_features=False, use_lp_ft_file=False):
    all_results = []
    summary_stats = []
    aggregated_results = {}
    dataset_name = "WESAD" if "WESAD" in base_path else "StressID"

    # --- Primary results ---
    for model in models:
        if model == "cnn" and not is_zero_shot:
            # Transfer learning: try each classifier head separately
            for head in CNN_HEADS:
                model_name = f"cnn_{head}"
                results = load_results(base_path, strategy_dir, model_name, SEEDS,
                                       use_lp_ft_file=use_lp_ft_file)
                process_model_results(results, model_name, all_results, summary_stats,
                                      aggregated_results, is_zero_shot)
        else:
            # Zero-shot CNN has no head — load as plain "cnn"
            results = load_results(base_path, strategy_dir, model, SEEDS,
                                   is_zero_shot=is_zero_shot)
            process_model_results(results, model, all_results, summary_stats,
                                  aggregated_results, is_zero_shot)

    # --- Feature-engineered baseline (lp_ft strategy only) ---
    if include_features and not is_zero_shot:
        for ws, ss in [(10, 5), (30, 10)]:
            feat_results = load_feature_results(base_path, dataset_name, ws, ss, SEEDS)
            if feat_results:
                model_name = f"feature_engineered_lr_{ws}_{ss}"
                process_model_results(feat_results, model_name, all_results, summary_stats,
                                      aggregated_results, is_zero_shot)

    # --- Save outputs ---
    suffix = "_zero_shot" if is_zero_shot else ""
    out_dir = os.path.join(base_path, strategy_dir)

    individual_csv = os.path.join(out_dir, f"individual_results_comparison{suffix}.csv")
    pd.DataFrame(all_results).to_csv(individual_csv, index=False)
    print(f"\nIndividual results  -> {individual_csv}")

    summary_csv = os.path.join(out_dir, f"summary_statistics_comparison{suffix}.csv")
    pd.DataFrame(summary_stats).to_csv(summary_csv, index=False)
    print(f"Summary statistics  -> {summary_csv}")

    if is_zero_shot and aggregated_results:
        json_out = os.path.join(out_dir, f"aggregated_zero_shot_results_{dataset_name}.json")
        with open(json_out, "w") as f:
            json.dump(aggregated_results, f, indent=2)
        print(f"Aggregated JSON     -> {json_out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate transfer-learning / zero-shot results across seeds."
    )
    parser.add_argument("--dataset", default="wesad", choices=("wesad", "stressid"))
    parser.add_argument(
        "--strategy", default="lp_ft",
        choices=("lp_ft", "head_only", "zero_shot"),
        help=(
            "lp_ft      = fine_tuned_encoder_new_head  (encoder fine-tuned + new head)\n"
            "head_only  = pretrained_encoder_new_head  (frozen encoder + new head)\n"
            "zero_shot  = zero_shot_performance"
        ),
    )
    args = parser.parse_args()

    SEEDS = [3, 5, 7, 9, 42]
    MODELS = ["cnn", "TSTCC"]
    CNN_HEADS = ["mlp", "logistic_regression"]

    dataset_name = "WESAD" if args.dataset == "wesad" else "StressID"
    is_zero_shot = args.strategy == "zero_shot"
    use_lp_ft_file = args.strategy == "lp_ft"
    include_features = args.strategy == "lp_ft"

    strategy_dir = STRATEGY_TO_DIR[args.strategy]
    base_path = os.path.join(RESULTS_PATH, "Transfer_learning", dataset_name)

    if is_zero_shot:
        models = ["cnn", "TSTCC", "feature_engineered_ws30_ss10", "feature_engineered_ws10_ss5"]
        print("=== Zero-Shot Results ===")
    else:
        models = MODELS
        print(f"=== Transfer Learning Results  [{args.strategy}] ===")
        print(f"Dataset: {dataset_name}  |  Label fraction: 1.0  |  Window: 10 / Step: 5")
        if use_lp_ft_file:
            print("Reading test_results_lp_ft.json (LP+FT).")
        if include_features:
            print("Including feature-engineered baseline.")

    main(
        base_path, strategy_dir, models,
        is_zero_shot=is_zero_shot,
        include_features=include_features,
        use_lp_ft_file=use_lp_ft_file,
    )

    # Important: By default, the feature engineered solution is also shown and saved when lp_ft is selected!