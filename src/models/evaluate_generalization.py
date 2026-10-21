"""
evaluate_generalization.py

Evaluates Cross-Project Generalization (RQ3):
Tests whether a ranking model trained on Cloudify can prioritize tests 
on Graylog without retraining, comparing raw duration vs. normalized duration.
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

    # 1. Train model strictly on Cloudify data
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
        "Cross-Project LTR (Raw Duration)": {"apfd": [], "apfdc": []},
        "Cross-Project LTR (Normalized Duration)": {"apfd": [], "apfdc": []}
    }

    for b_id in failing_graylog_builds:
        b_df = df_graylog[df_graylog["build_id"] == b_id].copy()
        if len(b_df) < 2:
            continue

        # Strategy 1: Random Baseline
        rand_df = b_df.sample(frac=1.0, random_state=42)
        results["Random Baseline"]["apfd"].append(compute_apfd(rand_df["target"].tolist()))
        results["Random Baseline"]["apfdc"].append(
            compute_apfdc(rand_df["target"].tolist(), rand_df["duration"].tolist())
        )

        pred_scores = ranker.predict(b_df[feature_cols])

        # Strategy 2: Cross-Project Raw Duration Division
        b_df["score_raw"] = pred_scores / (b_df["mean_duration_last5"] + 1e-4)
        raw_df = b_df.sort_values(by="score_raw", ascending=False)
        results["Cross-Project LTR (Raw Duration)"]["apfd"].append(
            compute_apfd(raw_df["target"].tolist())
        )
        results["Cross-Project LTR (Raw Duration)"]["apfdc"].append(
            compute_apfdc(raw_df["target"].tolist(), raw_df["duration"].tolist())
        )

        # Strategy 3: Cross-Project Normalized Duration (Within-build relative cost)
        dur_min = b_df["mean_duration_last5"].min()
        dur_max = b_df["mean_duration_last5"].max()
        if dur_max > dur_min:
            norm_dur = (b_df["mean_duration_last5"] - dur_min) / (dur_max - dur_min)
        else:
            norm_dur = np.zeros(len(b_df))
        
        # Relative cost weighting
        b_df["score_norm"] = pred_scores / (1.0 + norm_dur)
        norm_df = b_df.sort_values(by="score_norm", ascending=False)
        results["Cross-Project LTR (Normalized Duration)"]["apfd"].append(
            compute_apfd(norm_df["target"].tolist())
        )
        results["Cross-Project LTR (Normalized Duration)"]["apfdc"].append(
            compute_apfdc(norm_df["target"].tolist(), norm_df["duration"].tolist())
        )

    print("\n" + "=" * 82)
    print(f"{'EVALUATION STRATEGY (RQ3)':<55} | {'APFD':<8} | {'APFDc':<8}")
    print("=" * 82)
    for strat, scores in results.items():
        m_apfd = np.mean(scores["apfd"]) if len(scores["apfd"]) > 0 else 0.0
        m_apfdc = np.mean(scores["apfdc"]) if len(scores["apfdc"]) > 0 else 0.0
        print(f"{strat:<55} | {m_apfd:<8.4f} | {m_apfdc:<8.4f}")
    print("=" * 82)

if __name__ == "__main__":
    run_cross_project_evaluation()