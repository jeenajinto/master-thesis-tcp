"""metrics.py

Evaluation metrics for Test Case Prioritization: APFD, APFDc, and TTF.
"""

from typing import List
import numpy as np


def compute_apfd(prioritized_failures: List[int]) -> float:
  """Computes standard APFD.

  Args:
      prioritized_failures: Ordered binary list where 1 = test failed, 0 =
        passed.

  Returns:
      APFD score between 0.0 and 1.0 (higher is better).
  """
  n = len(prioritized_failures)
  # 1-based indices of failing tests in the prioritized list
  fail_positions = [i + 1 for i, status in enumerate(prioritized_failures) if status == 1]
  m = len(fail_positions)

  if m == 0 or n == 0:
    return 1.0  # No failures in build, trivial pass

  apfd = 1.0 - (sum(fail_positions) / (n * m)) + (1.0 / (2.0 * n))
  return float(apfd)


def compute_apfdc(
    prioritized_failures: List[int], prioritized_durations: List[float]
) -> float:
  """Computes Cost-Cognizant APFD (APFDc).

  Args:
      prioritized_failures: Ordered binary list where 1 = test failed, 0 =
        passed.
      prioritized_durations: Execution duration (seconds) matching the
        prioritized order.

  Returns:
      APFDc score between 0.0 and 1.0.
  """
  n = len(prioritized_failures)
  fail_indices = [
      i for i, status in enumerate(prioritized_failures) if status == 1
  ]
  m = len(fail_indices)

  if m == 0 or n == 0:
    return 1.0

  total_cost = sum(prioritized_durations)
  if total_cost == 0:
    return compute_apfd(prioritized_failures)

  # Cumulative costs up to each failure point
  numerator = 0.0
  for fi in fail_indices:
    cost_after_first_detect = sum(prioritized_durations[fi:])
    half_own_cost = 0.5 * prioritized_durations[fi]
    numerator += cost_after_first_detect - half_own_cost

  apfdc = numerator / (total_cost * m)
  return float(apfdc)


def compute_ttf(
    prioritized_failures: List[int], prioritized_durations: List[float]
) -> float:
  """Time-To-First-Failure (TTF): wall-clock seconds until the first failure."""
  elapsed = 0.0
  for status, duration in zip(prioritized_failures, prioritized_durations):
    elapsed += duration
    if status == 1:
      return elapsed
  return elapsed