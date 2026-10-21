"""evaluate_prioritizer.py

Trains a Learning-to-Rank LightGBM model and evaluates APFD/APFDc
against traditional CI/CD baseline heuristics.
"""

from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_apfd, compute_apfdc


def run_experiment(
    data_path: str = "data/processed/cloudify_features.parquet",
):
  print(f"Loading feature dataset from: {data_path} ...")
  df = pd.read_parquet(data_path)

  # Identify builds that contain at least one failure
  build_failure_counts = df.groupby("build_id")["target"].sum()
  failing_build_ids = set(build_failure_counts[build_failure_counts > 0].index)

  print(f"Total builds: {df['build_id'].nunique():,}")
  print(
      f"Failing builds with actionable defects: {len(failing_build_ids):,}"
  )

  # Sequential 80/20 train-test split by build_step
  unique_builds = df[["build_id", "build_step"]].drop_duplicates().sort_values("build_step")
  split_idx = int(len(unique_builds) * 0.8)
  train_build_ids = set(unique_builds.iloc[:split_idx]["build_id"])
  test_build_ids = set(unique_builds.iloc[split_idx:]["build_id"])

  train_df = df[df["build_id"].isin(train_build_ids)].copy()
  test_df = df[df["build_id"].isin(test_build_ids)].copy()

  # Group query lengths required for Learning-to-Rank
  train_df = train_df.sort_values(by=["build_step"]).reset_index(drop=True)
  train_groups = train_df.groupby("build_id", sort=False).size().to_numpy()

  feature_cols = [
      "mean_duration_last5",
      "hist_fail_rate_last5",
      "hist_fail_rate_last10",
      "consecutive_passes",
      "recency",
  ]

  X_train = train_df[feature_cols]
  y_train = train_df["target"]

  print("\n--> Training LightGBM LambdaMART ranker...")
  ranker = lgb.LGBMRanker(
      objective="lambdarank",
      n_estimators=100,
      learning_rate=0.05,
      random_state=42,
  )
  ranker.fit(
      X_train,
      y_train,
      group=train_groups,
  )

  # Evaluate on test builds that contain at least one failure
  test_eval_builds = [b for b in unique_builds.iloc[split_idx:]["build_id"] if b in failing_build_ids]
  print(f"--> Evaluating strategies on {len(test_eval_builds)} test builds with failures...")

  results = {
      "Random": {"apfd": [], "apfdc": []},
      "Fastest-First": {"apfd": [], "apfdc": []},
      "Recent-Failure-First (RFF)": {"apfd": [], "apfdc": []},
      "LightGBM (Cost-Aware LTR)": {"apfd": [], "apfdc": []},
  }

  for b_id in test_eval_builds:
    b_df = test_df[test_df["build_id"] == b_id].copy()
    if len(b_df) < 2:
      continue

    # 1. Baseline: Random Ordering
    rand_df = b_df.sample(frac=1.0, random_state=42)
    results["Random"]["apfd"].append(compute_apfd(rand_df["target"].tolist()))
    results["Random"]["apfdc"].append(
        compute_apfdc(
            rand_df["target"].tolist(), rand_df["duration"].tolist()
        )
    )

    # 2. Baseline: Fastest-First (Duration ascending)
    fast_df = b_df.sort_values(by="mean_duration_last5", ascending=True)
    results["Fastest-First"]["apfd"].append(
        compute_apfd(fast_df["target"].tolist())
    )
    results["Fastest-First"]["apfdc"].append(
        compute_apfdc(
            fast_df["target"].tolist(), fast_df["duration"].tolist()
        )
    )

    # 3. Baseline: Recent Failure First (RFF heuristic)
    rff_df = b_df.sort_values(by="hist_fail_rate_last5", ascending=False)
    results["Recent-Failure-First (RFF)"]["apfd"].append(
        compute_apfd(rff_df["target"].tolist())
    )
    results["Recent-Failure-First (RFF)"]["apfdc"].append(
        compute_apfdc(rff_df["target"].tolist(), rff_df["duration"].tolist())
    )

    # 4. LightGBM Cost-Aware Ranker: score = predicted_relevance / (duration + 1e-4)
    pred_scores = ranker.predict(b_df[feature_cols])
    b_df["rank_score"] = pred_scores / (b_df["mean_duration_last5"] + 1e-4)
    ml_df = b_df.sort_values(by="rank_score", ascending=False)
    results["LightGBM (Cost-Aware LTR)"]["apfd"].append(
        compute_apfd(ml_df["target"].tolist())
    )
    results["LightGBM (Cost-Aware LTR)"]["apfdc"].append(
        compute_apfdc(ml_df["target"].tolist(), ml_df["duration"].tolist())
    )

  print("\n" + "=" * 55)
  print(f"{'PRIORITIZATION STRATEGY':<30} | {'APFD':<8} | {'APFDc':<8}")
  print("=" * 55)
  for strat, scores in results.items():
    mean_apfd = np.mean(scores["apfd"]) if len(scores["apfd"]) > 0 else 0.0
    mean_apfdc = np.mean(scores["apfdc"]) if len(scores["apfdc"]) > 0 else 0.0
    print(f"{strat:<30} | {mean_apfd:<8.4f} | {mean_apfdc:<8.4f}")
  print("=" * 55)


if __name__ == "__main__":
  run_experiment()