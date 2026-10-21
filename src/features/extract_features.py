from pathlib import Path
import numpy as np
import pandas as pd


def compute_historical_features(
    input_path: str = "data/processed/cloudify_clean.csv",
    output_path: str = "data/processed/cloudify_features.parquet",
):
  print(f"Loading cleaned data from: {input_path}")
  df = pd.read_csv(input_path)

  # 1. Ensure temporal order by build_id
  # Builds must be chronological to prevent lookahead bias
  unique_builds = df["build_id"].drop_duplicates().tolist()
  build_order_map = {b_id: i for i, b_id in enumerate(unique_builds)}
  df["build_step"] = df["build_id"].map(build_order_map)
  df = df.sort_values(by=["build_step"]).reset_index(drop=True)

  print(f"Processing {len(unique_builds):,} unique CI builds...")

  # 2. Tracking test-level historical execution states
  test_history = {}
  # test_name -> {
  #     'past_statuses': list of int (0/1),
  #     'past_durations': list of float,
  #     'last_seen_build_step': int
  # }

  features = []

  for row in df.itertuples(index=False):
    t_name = row.test_name
    b_step = row.build_step
    status = row.status
    duration = row.duration

    if t_name not in test_history:
      # Cold start for test
      h_fail_5 = 0.0
      h_fail_10 = 0.0
      consec_passes = 0
      avg_dur_5 = float(duration)
      recency = 0
      test_history[t_name] = {
          "past_statuses": [],
          "past_durations": [],
          "last_seen_step": b_step,
      }
    else:
      hist = test_history[t_name]
      past_stat = hist["past_statuses"]
      past_dur = hist["past_durations"]

      # Rolling failure rates
      h_fail_5 = np.mean(past_stat[-5:]) if len(past_stat) > 0 else 0.0
      h_fail_10 = np.mean(past_stat[-10:]) if len(past_stat) > 0 else 0.0

      # Consecutive passes
      consec = 0
      for s in reversed(past_stat):
        if s == 0:
          consec += 1
        else:
          break
      consec_passes = consec

      # Moving average duration (cost)
      avg_dur_5 = (
          np.mean(past_dur[-5:]) if len(past_dur) > 0 else float(duration)
      )

      # Recency: builds elapsed since last execution
      recency = b_step - hist["last_seen_step"]

    # Append features for current test run
    features.append({
        "build_id": row.build_id,
        "build_step": b_step,
        "test_name": t_name,
        "duration": float(duration),
        "mean_duration_last5": float(avg_dur_5),
        "hist_fail_rate_last5": float(h_fail_5),
        "hist_fail_rate_last10": float(h_fail_10),
        "consecutive_passes": consec_passes,
        "recency": recency,
        "target": int(status),
    })

    # Update history AFTER feature generation (prevents data leakage)
    test_history[t_name]["past_statuses"].append(int(status))
    test_history[t_name]["past_durations"].append(float(duration))
    test_history[t_name]["last_seen_step"] = b_step

  features_df = pd.DataFrame(features)

  # 3. Save feature matrix
  out_file = Path(output_path)
  out_file.parent.mkdir(parents=True, exist_ok=True)
  features_df.to_parquet(out_file, index=False)

  print(f"\nFeature extraction complete! Saved to: {out_file}")
  print(f"Total instances extracted: {len(features_df):,}")
  print("\nFeature preview:")
  print(
      features_df[[
          "test_name",
          "hist_fail_rate_last5",
          "consecutive_passes",
          "mean_duration_last5",
          "target",
      ]].head(5)
  )


if __name__ == "__main__":
  compute_historical_features()