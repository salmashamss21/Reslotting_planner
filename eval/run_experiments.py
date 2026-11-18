# eval/run_experiments.py
# This script runs the three required P8 experiments (plus NN vs Baselines)
# across 5 random seeds [42, 43, 44, 45, 46] and saves all tables (CSV + Markdown)
# and plots (PNG with mean and standard deviation error bars) in results/.

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nn.demand_model import run_demand_forecasting
from csp.warehouse_csp import build_csp_context, solve_csp_single_distance, solve_csp_with_relaxation
from search.swap_search import greedy_best_first_search, bounded_ucs, astar_search
from integration.pipeline import load_prepared_data, evaluate_four_layouts_for_week

SEEDS = [42, 43, 44, 45, 46]


# Run NN vs Last-Week and 4-Week-Average baselines across 5 seeds.
def run_nn_baseline_experiment(bundle, results_dir="results"):
    rows = []
    cached_forecasts = {}

    for seed in SEEDS:
        metrics, test_records, _, _ = run_demand_forecasting(bundle["weekly_df"], seed=seed)
        cached_forecasts[seed] = test_records
        for m_name, key_v, key_t in [
            ("Last-Week", "val_last_week", "test_last_week"),
            ("4-Week Avg", "val_4week_avg", "test_4week_avg"),
            ("NN (MLP)", "val_nn", "test_nn")
        ]:
            rows.append({
                "seed": seed,
                "model": m_name,
                "val_mae": metrics[key_v][0],
                "val_rmse": metrics[key_v][1],
                "test_mae": metrics[key_t][0],
                "test_rmse": metrics[key_t][1]
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "nn_vs_baselines.csv"), index=False)

    summary = df.groupby("model", sort=False).agg(
        val_mae_mean=("val_mae", "mean"), val_mae_std=("val_mae", "std"),
        test_mae_mean=("test_mae", "mean"), test_mae_std=("test_mae", "std")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(summary))
    axes[0].bar(x, summary["val_mae_mean"], yerr=summary["val_mae_std"], capsize=5,
                color=["#94a3b8", "#60a5fa", "#2563eb"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["model"])
    axes[0].set_title("Validation Set Forecast MAE (Mean ± Std)")
    axes[0].set_ylabel("MAE (Units)")

    axes[1].bar(x, summary["test_mae_mean"], yerr=summary["test_mae_std"], capsize=5,
                color=["#94a3b8", "#60a5fa", "#2563eb"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["model"])
    axes[1].set_title("Test Set Forecast MAE (Mean ± Std)")
    axes[1].set_ylabel("MAE (Units)")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "nn_vs_baselines.png"), dpi=110)
    plt.close(fig)
    return df, cached_forecasts


# Experiment 1: Compare Greedy Best-First vs Bounded UCS vs A* across 5 seeds.
def run_experiment_1_search(bundle, cached_forecasts, results_dir="results"):
    prods = bundle["products_list"]
    slots = bundle["slots_list"]
    pairs = bundle["copurchase_pairs"]
    rows = []

    for seed in SEEDS:
        test_records = cached_forecasts[seed]
        test_weeks = sorted(list(set(int(r["week_index"]) for r in test_records)))
        rng = random.Random(seed)
        demand = {}
        for rec in test_records:
            if int(rec["week_index"]) == test_weeks[0]:
                noise = rng.uniform(0.95, 1.05)
                demand[str(rec["StockCode"])] = float(rec["pred_nn"]) * noise

        init_layout, csp_st = solve_csp_with_relaxation(prods, slots, pairs, demand, initial_max_distance=6)
        ctx = build_csp_context(prods, slots, pairs, csp_st["loosest_distance_used"])

        for algo_label, func in [
            ("Greedy Best-First", greedy_best_first_search),
            ("Bounded UCS", bounded_ucs),
            ("A* Variant", astar_search)
        ]:
            t0 = time.perf_counter()
            _, st = func(init_layout, demand, ctx, max_swaps=6)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            rows.append({
                "seed": seed,
                "algorithm": algo_label,
                "initial_distance": st["initial_distance"],
                "final_distance": st["final_distance"],
                "distance_saved": round(st["initial_distance"] - st["final_distance"], 2),
                "swaps": st["swaps_count"],
                "nodes_expanded": st["nodes_expanded"],
                "nodes_generated": st["nodes_generated"],
                "runtime_ms": round(elapsed_ms, 2)
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "exp1_search_comparison.csv"), index=False)

    summary = df.groupby("algorithm", sort=False).agg(
        final_mean=("final_distance", "mean"), final_std=("final_distance", "std"),
        saved_mean=("distance_saved", "mean"), saved_std=("distance_saved", "std"),
        exp_mean=("nodes_expanded", "mean"), exp_std=("nodes_expanded", "std"),
        gen_mean=("nodes_generated", "mean"), gen_std=("nodes_generated", "std")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(summary))
    axes[0].bar(x, summary["saved_mean"], yerr=summary["saved_std"], capsize=5,
                color=["#22c55e", "#f59e0b", "#3b82f6"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(summary["algorithm"])
    axes[0].set_title("Exp 1: Walking Distance Saved by Swaps (Mean ± Std)")
    axes[0].set_ylabel("Distance Saved")

    axes[1].bar(x - 0.18, summary["exp_mean"], yerr=summary["exp_std"], width=0.35,
                capsize=4, label="Nodes Expanded", color="#6366f1")
    axes[1].bar(x + 0.18, summary["gen_mean"], yerr=summary["gen_std"], width=0.35,
                capsize=4, label="Nodes Generated", color="#a855f7")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(summary["algorithm"])
    axes[1].set_title("Exp 1: Search Nodes Expanded & Generated (Mean ± Std)")
    axes[1].set_ylabel("Node Count")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp1_search_comparison.png"), dpi=110)
    plt.close(fig)
    return df


# Experiment 2: Compare BT vs BT+FC vs BT+FC+AC-3 for 60, 90, and 120 products across 5 seeds.
def run_experiment_2_csp_scaling(bundle, cached_forecasts, results_dir="results"):
    all_prods = bundle["products_list"]
    all_slots = bundle["slots_list"]
    pairs = bundle["copurchase_pairs"]
    rows = []

    for n_prod in [60, 90, 120]:
        sub_prods = all_prods[:n_prod]
        for seed in SEEDS:
            test_records = cached_forecasts[seed]
            test_weeks = sorted(list(set(int(r["week_index"]) for r in test_records)))
            rng = random.Random(seed * 100 + n_prod)
            demand = {}
            for rec in test_records:
                if int(rec["week_index"]) == test_weeks[0]:
                    demand[str(rec["StockCode"])] = float(rec["pred_nn"]) * rng.uniform(0.92, 1.08)

            for mode_name, use_fc, use_ac3 in [
                ("BT", False, False),
                ("BT+FC", True, False),
                ("BT+FC+AC3", True, True)
            ]:
                t0 = time.perf_counter()
                _, st = solve_csp_single_distance(
                    sub_prods, all_slots, pairs, demand,
                    max_distance=6, use_fc=use_fc, use_ac3=use_ac3, max_backtracks=25000
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                rows.append({
                    "num_products": n_prod,
                    "seed": seed,
                    "mode": mode_name,
                    "assignments": st["assignments"],
                    "backtracks": st["backtracks"],
                    "pruned_values": st["pruned_values"],
                    "runtime_ms": round(elapsed_ms, 2)
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "exp2_csp_scaling.csv"), index=False)

    summary = df.groupby(["num_products", "mode"], sort=False).agg(
        bt_mean=("backtracks", "mean"), bt_std=("backtracks", "std"),
        pruned_mean=("pruned_values", "mean"), pruned_std=("pruned_values", "std")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    sizes = [60, 90, 120]
    x = np.arange(len(sizes))
    colors = {"BT": "#ef4444", "BT+FC": "#f59e0b", "BT+FC+AC3": "#10b981"}

    for idx, mode in enumerate(["BT", "BT+FC", "BT+FC+AC3"]):
        sub = summary[summary["mode"] == mode]
        axes[0].bar(x + (idx - 1) * 0.25, sub["bt_mean"], yerr=sub["bt_std"],
                    width=0.24, capsize=4, label=mode, color=colors[mode])
        axes[1].bar(x + (idx - 1) * 0.25, sub["pruned_mean"], yerr=sub["pruned_std"],
                    width=0.24, capsize=4, label=mode, color=colors[mode])

    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{s} Products" for s in sizes])
    axes[0].set_title("Exp 2: CSP Backtracks by Size (Mean ± Std)")
    axes[0].set_ylabel("Backtracks")
    axes[0].legend()

    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{s} Products" for s in sizes])
    axes[1].set_title("Exp 2: Domain Values Pruned by Size (Mean ± Std)")
    axes[1].set_ylabel("Pruned Slot Values")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp2_csp_scaling.png"), dpi=110)
    plt.close(fig)
    return df


# Experiment 3: Compare Picker Walking Distance over 10 Test Weeks for the 4 Layouts across 5 seeds.
def run_experiment_3_four_layouts(bundle, cached_forecasts, results_dir="results"):
    rows = []
    for seed in SEEDS:
        test_records = cached_forecasts[seed]
        test_weeks = sorted(list(set(int(r["week_index"]) for r in test_records)))
        for wk in test_weeks:
            res = evaluate_four_layouts_for_week(
                wk, test_records, bundle, initial_max_dist=6, max_swaps=6, seed=seed
            )
            rows.append({
                "seed": seed,
                "week_index": wk,
                "num_orders": res["num_orders"],
                "Random": res["random_distance"],
                "Last_Week": res["last_week_distance"],
                "NN_Demand": res["nn_distance"],
                "NN_Plus_Swaps": res["nn_swaps_distance"]
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "exp3_four_layouts_10weeks.csv"), index=False)

    wk_summary = df.groupby("week_index").agg(
        rand_m=("Random", "mean"), rand_s=("Random", "std"),
        lw_m=("Last_Week", "mean"), lw_s=("Last_Week", "std"),
        nn_m=("NN_Demand", "mean"), nn_s=("NN_Demand", "std"),
        sw_m=("NN_Plus_Swaps", "mean"), sw_s=("NN_Plus_Swaps", "std")
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    weeks = wk_summary["week_index"]
    axes[0].plot(weeks, wk_summary["rand_m"], marker="s", label="1. Random", color="#94a3b8")
    axes[0].plot(weeks, wk_summary["lw_m"], marker="^", label="2. Last-Week Demand", color="#f59e0b")
    axes[0].plot(weeks, wk_summary["nn_m"], marker="o", label="3. NN Demand", color="#3b82f6")
    axes[0].plot(weeks, wk_summary["sw_m"], marker="D", label="4. NN + Swaps", color="#16a34a")
    axes[0].set_title("Exp 3: Weekly Picker Distance over 10 Test Weeks")
    axes[0].set_xlabel("Test Week Index (Nov–Dec 2011)")
    axes[0].set_ylabel("Total Picker Walking Distance (m)")
    axes[0].legend()

    seed_totals = df.groupby("seed")[["Random", "Last_Week", "NN_Demand", "NN_Plus_Swaps"]].sum()
    means = [seed_totals[c].mean() for c in ["Random", "Last_Week", "NN_Demand", "NN_Plus_Swaps"]]
    stds = [seed_totals[c].std() for c in ["Random", "Last_Week", "NN_Demand", "NN_Plus_Swaps"]]
    labels = ["Random", "Last-Week", "NN Demand", "NN + Swaps"]
    x = np.arange(4)
    axes[1].bar(x, means, yerr=stds, capsize=5, color=["#cbd5e1", "#fbbf24", "#38bdf8", "#22c55e"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_title("Exp 3: Total 10-Week Picker Distance (Mean ± Std)")
    axes[1].set_ylabel("Cumulative 10-Week Distance (m)")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "exp3_four_layouts_10weeks.png"), dpi=110)
    plt.close(fig)
    return df


# Write a formatted Markdown summary of all experiments (Mean +- Std) to results/experiments_summary.md.
def save_markdown_summary(nn_df, exp1_df, exp2_df, exp3_df, results_dir="results"):
    lines = ["# P8 Experimental Results Summary (5 Seeds: 42, 43, 44, 45, 46)\n"]

    lines.append("## 0. Demand Forecasting: Neural Network vs. Baselines")
    lines.append("| Model | Val MAE (Mean ± Std) | Val RMSE (Mean ± Std) | Test MAE (Mean ± Std) | Test RMSE (Mean ± Std) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: |")
    for m, grp in nn_df.groupby("model", sort=False):
        lines.append(
            f"| **{m}** | {grp['val_mae'].mean():.2f} ± {grp['val_mae'].std():.2f} | "
            f"{grp['val_rmse'].mean():.2f} ± {grp['val_rmse'].std():.2f} | "
            f"{grp['test_mae'].mean():.2f} ± {grp['test_mae'].std():.2f} | "
            f"{grp['test_rmse'].mean():.2f} ± {grp['test_rmse'].std():.2f} |"
        )

    lines.append("\n## Experiment 1: Search Algorithms (Greedy Best-First vs. Bounded UCS vs. A*)")
    lines.append("| Algorithm | Initial Cost | Final Cost (Mean ± Std) | Distance Saved (Mean ± Std) | Swaps | Nodes Expanded | Nodes Generated | Runtime (ms) |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for alg, grp in exp1_df.groupby("algorithm", sort=False):
        lines.append(
            f"| **{alg}** | {grp['initial_distance'].mean():.1f} | "
            f"{grp['final_distance'].mean():.1f} ± {grp['final_distance'].std():.1f} | "
            f"{grp['distance_saved'].mean():.1f} ± {grp['distance_saved'].std():.1f} | "
            f"{grp['swaps'].mean():.1f} ± {grp['swaps'].std():.1f} | "
            f"{grp['nodes_expanded'].mean():.1f} ± {grp['nodes_expanded'].std():.1f} | "
            f"{grp['nodes_generated'].mean():.1f} ± {grp['nodes_generated'].std():.1f} | "
            f"{grp['runtime_ms'].mean():.2f} ± {grp['runtime_ms'].std():.2f} |"
        )

    lines.append("\n## Experiment 2: CSP Solver Scaling (BT vs. BT+FC vs. BT+FC+AC-3 for 60, 90, 120 Products)")
    lines.append("| Products | CSP Mode | Assignments (Mean ± Std) | Backtracks (Mean ± Std) | Values Pruned (Mean ± Std) | Runtime (ms) |")
    lines.append("| :---: | :--- | :---: | :---: | :---: | :---: |")
    for (n_p, mode), grp in exp2_df.groupby(["num_products", "mode"], sort=False):
        lines.append(
            f"| {n_p} | **{mode}** | {grp['assignments'].mean():.1f} ± {grp['assignments'].std():.1f} | "
            f"{grp['backtracks'].mean():.1f} ± {grp['backtracks'].std():.1f} | "
            f"{grp['pruned_values'].mean():.1f} ± {grp['pruned_values'].std():.1f} | "
            f"{grp['runtime_ms'].mean():.2f} ± {grp['runtime_ms'].std():.2f} |"
        )

    lines.append("\n## Experiment 3: Four-Layout Picker Walking Distance Over 10 Test Weeks")
    lines.append("| Week Index | Real Invoices | Random Layout (m) | Last-Week Layout (m) | NN Demand Layout (m) | NN + Swaps Layout (m) |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    for wk, grp in exp3_df.groupby("week_index"):
        n_inv = int(grp["num_orders"].iloc[0])
        lines.append(
            f"| Week {wk} | {n_inv} | {grp['Random'].mean():.1f} ± {grp['Random'].std():.1f} | "
            f"{grp['Last_Week'].mean():.1f} ± {grp['Last_Week'].std():.1f} | "
            f"{grp['NN_Demand'].mean():.1f} ± {grp['NN_Demand'].std():.1f} | "
            f"{grp['NN_Plus_Swaps'].mean():.1f} ± {grp['NN_Plus_Swaps'].std():.1f} |"
        )

    seed_sums = exp3_df.groupby("seed")[["Random", "Last_Week", "NN_Demand", "NN_Plus_Swaps"]].sum()
    lines.append(
        f"| **10-Week Total** | **{int(exp3_df.groupby('week_index')['num_orders'].first().sum())}** | "
        f"**{seed_sums['Random'].mean():.1f} ± {seed_sums['Random'].std():.1f}** | "
        f"**{seed_sums['Last_Week'].mean():.1f} ± {seed_sums['Last_Week'].std():.1f}** | "
        f"**{seed_sums['NN_Demand'].mean():.1f} ± {seed_sums['NN_Demand'].std():.1f}** | "
        f"**{seed_sums['NN_Plus_Swaps'].mean():.1f} ± {seed_sums['NN_Plus_Swaps'].std():.1f}** |"
    )

    out_md = os.path.join(results_dir, "experiments_summary.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("Saved formatted summary tables to:", out_md)


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    bundle = load_prepared_data("data")

    print("Running NN vs Baselines across 5 seeds...")
    nn_df, cached_forecasts = run_nn_baseline_experiment(bundle, results_dir)

    print("Running Experiment 1 (Greedy vs Bounded UCS vs A*)...")
    exp1_df = run_experiment_1_search(bundle, cached_forecasts, results_dir)

    print("Running Experiment 2 (BT vs BT+FC vs BT+FC+AC3 for 60, 90, 120 products)...")
    exp2_df = run_experiment_2_csp_scaling(bundle, cached_forecasts, results_dir)

    print("Running Experiment 3 (4 Layouts over 10 Test Weeks across 5 seeds)...")
    exp3_df = run_experiment_3_four_layouts(bundle, cached_forecasts, results_dir)

    save_markdown_summary(nn_df, exp1_df, exp2_df, exp3_df, results_dir)
    print("All experiments completed successfully!")


if __name__ == "__main__":
    main()
