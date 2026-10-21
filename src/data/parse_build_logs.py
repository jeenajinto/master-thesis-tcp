from pathlib import Path
import pandas as pd


def parse_cloudify_dataset(csv_path: str):
  print(f"Loading raw dataset from: {csv_path} ...")
  df = pd.read_csv(csv_path, low_memory=False)

  # Map the exact columns visible in the file
  # travisJobId -> build_id, testName -> test_name, duration -> duration
  df = df.rename(
      columns={
          "travisJobId": "build_id",
          "testName": "test_name",
          "duration": "duration",
      }
  )

  # Define binary failure target: 1 if failures > 0 or errors > 0, else 0
  df["status"] = (
      ((df["failures"] > 0) | (df["errors"] > 0)).astype(int)
      if ("failures" in df.columns and "errors" in df.columns)
      else 0
  )

  # Clean invalid durations and drop NaNs
  df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0.0)
  df_clean = df[["build_id", "test_name", "duration", "status"]].dropna()
  df_clean = df_clean[df_clean["duration"] >= 0]

  # Display summary stats
  total_runs = len(df_clean)
  total_builds = df_clean["build_id"].nunique()
  total_tests = df_clean["test_name"].nunique()
  total_fails = (df_clean["status"] == 1).sum()

  print("\n" + "=" * 45)
  print("        DATA SUMMARY VERIFIED")
  print("=" * 45)
  print(f"Total Test Runs   : {total_runs:,}")
  print(f"Total CI Builds   : {total_builds:,}")
  print(f"Unique Test Cases : {total_tests:,}")
  print(f"Failed Runs       : {total_fails:,}")
  print(f"Failure Rate      : {(total_fails / total_runs) * 100:.2f}%")
  print("=" * 45)

  # Save cleaned data
  out_dir = Path("data/processed")
  out_dir.mkdir(parents=True, exist_ok=True)
  out_path = out_dir / "cloudify_clean.csv"
  df_clean.to_csv(out_path, index=False)
  print(f"\nSuccessfully cleaned and saved to: {out_path}")
  return df_clean


if __name__ == "__main__":
  parse_cloudify_dataset("data/raw/cloudify.csv")