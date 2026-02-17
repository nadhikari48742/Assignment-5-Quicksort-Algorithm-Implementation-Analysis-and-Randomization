#!/usr/bin/env python3
"""
Assignment 5: Quicksort Algorithm: Implementation, Analysis, and Randomization
- Deterministic Quicksort using fixed pivot rule
- Randomized Quicksort using random pivot
- Empirical timing on Random/Sorted/Reverse-Sorted inputs
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt


# Quicksort Implementations

def partition(arr: List[int], lo: int, hi: int, pivot_index: int) -> int:
    """
    Lomuto partition scheme with explicit pivot_index.
    """
    pivot_value = arr[pivot_index]
    arr[pivot_index], arr[hi] = arr[hi], arr[pivot_index]  # move pivot to end

    store = lo
    for i in range(lo, hi):
        if arr[i] < pivot_value:
            arr[store], arr[i] = arr[i], arr[store]
            store += 1

    arr[store], arr[hi] = arr[hi], arr[store]  # move pivot to its final place
    return store


def quicksort_deterministic(arr: List[int], lo: int = 0, hi: int | None = None) -> None:
    """
    Deterministic in-place Quicksort.
    """
    if hi is None:
        hi = len(arr) - 1
    if lo >= hi:
        return

    pivot_index = (lo + hi) // 2
    p = partition(arr, lo, hi, pivot_index)
    quicksort_deterministic(arr, lo, p - 1)
    quicksort_deterministic(arr, p + 1, hi)


def quicksort_randomized(arr: List[int], rng: random.Random, lo: int = 0, hi: int | None = None) -> None:
    """
    Randomized in-place Quicksort.
    """
    if hi is None:
        hi = len(arr) - 1
    if lo >= hi:
        return

    pivot_index = rng.randint(lo, hi)
    p = partition(arr, lo, hi, pivot_index)
    quicksort_randomized(arr, rng, lo, p - 1)
    quicksort_randomized(arr, rng, p + 1, hi)


# Data generation

def gen_random(n: int, rng: random.Random, value_range: Tuple[int, int] = (0, 100000)) -> List[int]:
    a, b = value_range
    return [rng.randint(a, b) for _ in range(n)]


def gen_sorted(n: int) -> List[int]:
    return list(range(n))


def gen_reverse_sorted(n: int) -> List[int]:
    return list(range(n, 0, -1))


# Benchmarking

def time_sort(sort_fn: Callable[[], None], trials: int) -> float:
    """
    Returns average time over 'trials' runs using perf_counter().
    """
    total = 0.0
    for _ in range(trials):
        t0 = time.perf_counter()
        sort_fn()
        total += (time.perf_counter() - t0)
    return total / trials


@dataclass
class BenchConfig:
    sizes: List[int]
    trials: int
    seed: int
    outdir: str


def ensure_outdir(outdir: str) -> None:
    os.makedirs(outdir, exist_ok=True)


def benchmark(cfg: BenchConfig) -> Dict[str, Dict[str, List[float]]]:
    """
    Returns results[algorithm][distribution] = list of avg times aligned with cfg.sizes
    """
    rng_master = random.Random(cfg.seed)

    distributions: Dict[str, Callable[[int], List[int]]] = {
        "Random": lambda n: gen_random(n, rng_master),
        "Sorted": gen_sorted,
        "Reverse Sorted": gen_reverse_sorted,
    }

    results: Dict[str, Dict[str, List[float]]] = {
        "Deterministic": {name: [] for name in distributions},
        "Randomized": {name: [] for name in distributions},
    }

    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))

    for dist_name, make_arr in distributions.items():
        for n in cfg.sizes:
            base = make_arr(n)

            # Deterministic benchmark
            def run_det() -> None:
                arr = base.copy()
                quicksort_deterministic(arr)

            t_det = time_sort(run_det, cfg.trials)
            results["Deterministic"][dist_name].append(t_det)

            # Randomized benchmark
            def run_rand() -> None:
                arr = base.copy()
                rng = random.Random(cfg.seed + n)
                quicksort_randomized(arr, rng)

            t_rand = time_sort(run_rand, cfg.trials)
            results["Randomized"][dist_name].append(t_rand)

            print(f"[{dist_name:14}] n={n:6}  det={t_det:.6f}s  rand={t_rand:.6f}s")

    return results


# Output CSV and Plots

def write_csv(cfg: BenchConfig, results: Dict[str, Dict[str, List[float]]]) -> str:
    path = os.path.join(cfg.outdir, "quicksort_results.csv")
    dists = list(results["Deterministic"].keys())

    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["distribution", "n", "deterministic_avg_s", "randomized_avg_s"])
        for dist in dists:
            for i, n in enumerate(cfg.sizes):
                w.writerow([dist, n, results["Deterministic"][dist][i], results["Randomized"][dist][i]])

    return path


def plot_results(cfg: BenchConfig, results: Dict[str, Dict[str, List[float]]]) -> List[str]:
    paths: List[str] = []
    for dist in results["Deterministic"].keys():
        plt.figure(figsize=(10, 5))
        plt.plot(cfg.sizes, results["Deterministic"][dist], label="Deterministic")
        plt.plot(cfg.sizes, results["Randomized"][dist], label="Randomized")
        plt.title(f"Performance Comparison ({dist} Distribution)")
        plt.xlabel("Input Size (n)")
        plt.ylabel("Average Execution Time (s)")
        plt.grid(True)
        plt.legend()

        outpath = os.path.join(cfg.outdir, f"plot_{dist.lower().replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
        paths.append(outpath)

    return paths


# Summary

def print_theory() -> None:
    print("\nTheoretical Analysis (Quicksort):")
    print("- Best case:    O(n log n)  (balanced partitions each level)")
    print("- Average case: O(n log n)  (expected balanced split over inputs/pivots)")
    print("- Worst case:   O(n^2)      (highly unbalanced partitions, e.g., 0 and n-1)")
    print("- Space:        O(log n) average recursion depth; O(n) worst-case recursion depth")
    print("\nRandomization effect:")
    print("- Random pivot makes worst-case partitions unlikely on any fixed input,")
    print("  so expected running time remains O(n log n) and avoids predictable bad cases.")


# Main

def parse_args() -> BenchConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-size", type=int, default=20000, help="Largest input size to test")
    ap.add_argument("--sizes", type=str, default="", help="Comma-separated sizes (overrides --max-size)")
    ap.add_argument("--trials", type=int, default=5, help="Trials per (distribution, size)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    ap.add_argument("--outdir", type=str, default="results", help="Output directory for CSV and plots")
    
    args = ap.parse_args([])

    if args.sizes.strip():
        sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    else:
        # sensible default ladder up to max-size
        sizes = [100, 1000, 5000, 10000, args.max_size]

    return BenchConfig(sizes=sizes, trials=args.trials, seed=args.seed, outdir=args.outdir)


def main() -> None:
    cfg = parse_args()
    ensure_outdir(cfg.outdir)

    print("Running empirical analysis...")
    print(f"Sizes={cfg.sizes} | trials={cfg.trials} | seed={cfg.seed} | outdir='{cfg.outdir}'\n")

    results = benchmark(cfg)

    csv_path = write_csv(cfg, results)
    plot_paths = plot_results(cfg, results)

    print(f"\nSaved CSV:  {csv_path}")
    for p in plot_paths:
        print(f"Saved plot: {p}")

    print_theory()


if __name__ == "__main__":
    main()
