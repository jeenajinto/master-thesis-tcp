"""
evaluate_generalization.py

Evaluates Cross-Project Generalization (RQ3):
Tests whether a ranking model trained on Cloudify
can prioritize tests on Graylog without retraining.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import compute_apfd, compute_apfdc

def run_cross_project_evaluation():
    print("Loading Cloudify and Graylog feature sets...")
    df_cloudify = pd.read_parquet("data/processed/cloudify_features.parquet")
    df_graylog = pd.read_parquet("data/processed/graylog_features.parquet")

    feature_cols = [
        "mean_duration_last5",
        "hist_fail_rate_last5",
        "hist_fail_rate_last10",
        "consecutive_passes",
        "recency"
    ]

    # 1. Train Model strictly on Cloudify data
    print("Training LightGBM model on Cloudify...")
    df_cloudify = df_cloudify.sort_values(by=["build_step"]).reset_index(drop=True)
    train_groups = df_cloudify.groupby("build_id", sort=False).size().to_numpy()
    
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )
    ranker.fit(df_cloudify[feature_cols], df_cloudify["target"], group=train_groups)

    # 2. Identify failing builds in Graylog to evaluate
    build_fail_counts = df_graylog.groupby("build_id")["target"].sum()
    failing_graylog_builds = list(build_fail_counts[build_fail_counts > 0].index)
    print(f"Evaluating transfer performance on {len(failing_graylog_builds)} failing builds in Graylog...")

    results = {
        "Random Baseline": {"apfd": [], "apfdc": []},
        "Cross-Project LTR (Trained Cloudify -> Tested Graylog)": {"apfd": [], "apfdc": []}
    }

    for b_id in failing_graylog_builds:
        b_df = df_graylog[df_graylog["build_id"] == b_id].copy()
        if len(b_df) < 2:
            continue

        # Baseline: Random
        rand_df = b_df.sample(frac=1.0, random_state=42)
        results["Random Baseline"]["apfd"].append(compute_apfd(rand_df["target"].tolist()))
        results["Random Baseline"]["apfdc"].append(
            compute_apfdc(rand_df["target"].tolist(), rand_df["duration"].tolist())
        )

        # Proposed: Cross-Project ML Ranking
        pred_scores = ranker.predict(b_df[feature_cols])
        b_df["rank_score"] = pred_scores / (b_df["mean_duration_last5"] + 1e-4)
        ml_df = b_df.sort_values(by="rank_score", ascending=False)
        results["Cross-Project LTR (Trained Cloudify -> Tested Graylog)"]["apfd"].append(
            compute_apfd(ml_df["target"].tolist())
        )
        results["Cross-Project LTR (Trained Cloudify -> Tested Graylog)"]["apfdc"].append(
            compute_apfdc(ml_df["target"].tolist(), ml_df["duration"].tolist())
        )

    print("\n" + "=" * 78)
    print(f"{'EVALUATION STRATEGY (RQ3)':<55} | {'APFD':<8} | {'APFDc':<8}")
    print("=" * 78)
    for strat, scores in results.items():
        m_apfd = np.mean(scores["apfd"]) if len(scores["apfd"]) > 0 else 0.0
        m_apfdc = np.mean(scores["apfdc"]) if len(scores["apfdc"]) > 0 else 0.0
        print(f"{strat:<55} | {m_apfd:<8.4f} | {m_apfdc:<8.4f}")
    print("=" * 78)

if __name__ == "__main__":
    run_cross_project_evaluation()