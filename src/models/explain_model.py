"""
explain_model.py

Computes SHAP feature importance for the LightGBM ranker
to analyze which features drive prioritization decisions (RQ2).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt

# Ensure root directory is accessible
sys.path.append(str(Path(__file__).resolve().parents[2]))

def run_shap_analysis(data_path: str = "data/processed/cloudify_features.parquet"):
    print(f"Loading data from: {data_path} ...")
    df = pd.read_parquet(data_path)

    # Chronological 80/20 train/test split
    unique_builds = df[["build_id", "build_step"]].drop_duplicates().sort_values("build_step")
    split_idx = int(len(unique_builds) * 0.8)
    train_build_ids = set(unique_builds.iloc[:split_idx]["build_id"])
    test_build_ids = set(unique_builds.iloc[split_idx:]["build_id"])

    train_df = df[df["build_id"].isin(train_build_ids)].sort_values("build_step").reset_index(drop=True)
    test_df = df[df["build_id"].isin(test_build_ids)].sort_values("build_step").reset_index(drop=True)

    feature_cols = [
        "mean_duration_last5",
        "hist_fail_rate_last5",
        "hist_fail_rate_last10",
        "consecutive_passes",
        "recency"
    ]

    train_groups = train_df.groupby("build_id", sort=False).size().to_numpy()
    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    print("Training LightGBM model for TreeSHAP...")
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )
    ranker.fit(X_train, y_train, group=train_groups)

    # Sample a representative subset of test data for SHAP computation
    # (Evaluating 2,000 instances runs in seconds while remaining statistically accurate)
    X_test_sample = test_df[feature_cols].sample(n=min(2000, len(test_df)), random_state=42)

    print("Calculating TreeSHAP values...")
    explainer = shap.TreeExplainer(ranker)
    shap_values = explainer.shap_values(X_test_sample)

    # 1. Print mean absolute SHAP values (Global Ranking of features)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Mean |SHAP|": mean_abs_shap
    }).sort_values(by="Mean |SHAP|", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 45)
    print("      FEATURE IMPORTANCE RANKING (SHAP)")
    print("=" * 45)
    for idx, row in importance_df.iterrows():
        print(f"{idx+1}. {row['Feature']:<25} : {row['Mean |SHAP|']:.6f}")
    print("=" * 45)

    # 2. Save SHAP Summary Beeswarm plot to disk
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_path = figures_dir / "shap_summary.png"

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.title("SHAP Feature Importance for Test Case Prioritization", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\nSHAP summary plot saved to: {plot_path}")

if __name__ == "__main__":
    run_shap_analysis()