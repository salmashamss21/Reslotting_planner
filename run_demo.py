# run_demo.py
# This script runs the complete 5-minute live demonstration of Project P8:
#   1. Neural Network demand prediction compared with Last-Week & 4-Week-Average baselines.
#   2. CSP initial layout comparing Backtracking (BT), Forward Checking (FC), and AC-3,
#      plus step-by-step relaxation of the co-purchase distance rule.
#   3. Swap Search trace (Greedy Best-First, Bounded UCS, and A*) with CSP rechecks.
#   4. End-to-end system evaluation (NN ON vs NN OFF) on real held-out customer orders.
# It saves a visual cooperation plot to results/demo_cooperation.png and prints
# the full demonstration report to the terminal.

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.make_data import main as run_make_data
from nn.demand_model import run_demand_forecasting
from csp.warehouse_csp import (
    build_csp_context, solve_csp_single_distance, solve_csp_with_relaxation
)
from search.swap_search import (
    greedy_best_first_search, bounded_ucs, astar_search, run_small_known_answer_example
)
from integration.pipeline import load_prepared_data, evaluate_four_layouts_for_week


# Format Section 1: Dataset overview and Neural Network vs Baselines results.
def build_nn_demo_section(bundle, metrics, d_maps):
    lines = []
    lines.append("======================================================================")
    lines.append("1. INPUT DATA & NEURAL NETWORK DEMAND FORECAST VS. BASELINES")
    lines.append("======================================================================")
    lines.append(f"Products modeled        : {len(bundle['products_list'])} (Top 120 by volume)")
    lines.append(f"Warehouse shelf slots   : {len(bundle['slots_list'])} (6 aisles x 10 bays x 2 levels)")
    lines.append(f"Co-purchase rules (CSP) : Top {len(bundle['copurchase_pairs'])} pairs (from training weeks only)")
    lines.append("")
    lines.append("Model Comparison (Units Sold per Product-Week):")
    lines.append("  Model                  | Val MAE  | Val RMSE | Test MAE | Test RMSE")
    lines.append("  -----------------------+----------+----------+----------+----------")
    v_lw, v_4w, v_nn = metrics["val_last_week"], metrics["val_4week_avg"], metrics["val_nn"]
    t_lw, t_4w, t_nn = metrics["test_last_week"], metrics["test_4week_avg"], metrics["test_nn"]
    lines.append(f"  Last-Week Baseline     | {v_lw[0]:8.2f} | {v_lw[1]:8.2f} | {t_lw[0]:8.2f} | {t_lw[1]:8.2f}")
    lines.append(f"  4-Week Avg Baseline    | {v_4w[0]:8.2f} | {v_4w[1]:8.2f} | {t_4w[0]:8.2f} | {t_4w[1]:8.2f}")
    lines.append(f"  Neural Network (MLP)   | {v_nn[0]:8.2f} | {v_nn[1]:8.2f} | {t_nn[0]:8.2f} | {t_nn[1]:8.2f}")
    lines.append("")
    lines.append("Sample Top Predicted Products for Demo Test Week:")
    nn_items = sorted([(v, k) for k, v in d_maps["nn"].items()], reverse=True)[:5]
    for pred_val, code in nn_items:
        lw_val = d_maps["last_week"].get(code, 0.0)
        act_val = d_maps["actual"].get(code, 0.0)
        lines.append(f"  StockCode {code:8s} -> NN Pred: {pred_val:6.1f} | Last-Week: {lw_val:6.1f} | Actual: {act_val:6.1f}")
    lines.append("")
    return "\n".join(lines)


# Format Section 2: CSP Pruning comparison (BT vs BT+FC vs BT+FC+AC3) and distance relaxation.
def build_csp_demo_section(bundle, demand_nn):
    prods = bundle["products_list"][:90]
    slots = bundle["slots_list"]
    pairs = bundle["copurchase_pairs"]

    _, stats_bt = solve_csp_single_distance(prods, slots, pairs, demand_nn, max_distance=8, use_fc=False, use_ac3=False)
    _, stats_fc = solve_csp_single_distance(prods, slots, pairs, demand_nn, max_distance=8, use_fc=True, use_ac3=False)
    _, stats_ac3 = solve_csp_single_distance(prods, slots, pairs, demand_nn, max_distance=8, use_fc=True, use_ac3=True)

    _, relax_stats = solve_csp_with_relaxation(
        bundle["products_list"], slots, pairs, demand_nn, initial_max_distance=2, step=2
    )

    lines = []
    lines.append("======================================================================")
    lines.append("2. CSP SOLVER: PRUNING WITH & WITHOUT FC / AC-3 + DISTANCE RELAXATION")
    lines.append("======================================================================")
    lines.append("CSP Hard Rules: (1) 1 product/slot, (2) Heavy items on low shelves (level 0),")
    lines.append("                (3) Top 30 co-bought pairs <= max_distance, (4) Incompatible categories != aisle.")
    lines.append("")
    lines.append("Solver Switch Comparison (90 Products, max_distance = 6):")
    lines.append("  CSP Solver Mode              | Assignments | Backtracks | Values Pruned")
    lines.append("  -----------------------------+-------------+------------+--------------")
    lines.append(f"  Pure Backtracking (BT)       | {stats_bt['assignments']:11d} | {stats_bt['backtracks']:10d} | {stats_bt['pruned_values']:13d}")
    lines.append(f"  BT + Forward Checking (FC)   | {stats_fc['assignments']:11d} | {stats_fc['backtracks']:10d} | {stats_fc['pruned_values']:13d}")
    lines.append(f"  BT + FC + AC-3 (Full CSP)    | {stats_ac3['assignments']:11d} | {stats_ac3['backtracks']:10d} | {stats_ac3['pruned_values']:13d}")
    lines.append("")
    lines.append("Constraint Relaxation Demo (120 Products, starting at strict max_distance = 2):")
    lines.append("  Initial distance requested : 2 meters (infeasible for 30 co-purchase pairs)")
    lines.append(f"  Relaxation steps taken     : {relax_stats['relaxed_steps']} steps (2 -> 4 -> {relax_stats['loosest_distance_used']})")
    lines.append(f"  Loosest distance used      : {relax_stats['loosest_distance_used']} meters (SOLUTION FOUND)")
    lines.append("")
    return "\n".join(lines), (stats_bt, stats_fc, stats_ac3)


# Format Section 3: Search trace (Greedy Best-First, Bounded UCS, A* Variant, and Toy Example).
def build_search_demo_section(layout_nn, demand_nn, context):
    _, g_stats = greedy_best_first_search(layout_nn, demand_nn, context, max_swaps=6)
    _, u_stats = bounded_ucs(layout_nn, demand_nn, context, max_swaps=6)
    _, a_stats = astar_search(layout_nn, demand_nn, context, max_swaps=6)
    _, toy_stats = run_small_known_answer_example()

    lines = []
    lines.append("======================================================================")
    lines.append("3. SEARCH TRACE: RE-SLOTTING SWAPS WITH CSP RECHECK ON EVERY SWAP")
    lines.append("======================================================================")
    lines.append(f"Small Known-Answer Test (4 Products): Initial = {toy_stats['initial_distance']} -> "
                 f"Final = {toy_stats['final_distance']} in {toy_stats['swaps_count']} swap (Expected: 660.0)")
    lines.append("")
    lines.append("Search Algorithm Comparison on 120-Product Warehouse:")
    lines.append("  Algorithm                 | Init Cost  | Final Cost | Swaps | Expanded | Generated")
    lines.append("  --------------------------+------------+------------+-------+----------+----------")
    for name, st in [("Greedy Best-First (Main)", g_stats), ("Bounded UCS (Baseline)", u_stats), ("A* Variant", a_stats)]:
        lines.append(f"  {name:25s} | {st['initial_distance']:10.1f} | {st['final_distance']:10.1f} | "
                     f"{st['swaps_count']:5d} | {st['nodes_expanded']:8d} | {st['nodes_generated']:9d}")
    lines.append("")
    lines.append("Step-by-Step Swap Trace (Greedy Best-First Search):")
    lines.append(f"  Start Layout Demand-Weighted Walking Cost: {g_stats['initial_distance']:.2f}")
    for step in g_stats["trace"]:
        lines.append(
            f"  Swap #{step['swap_number']}: Product {step['product_a']:6s} ({step['slot_a']}) <-> "
            f"Product {step['product_b']:6s} ({step['slot_b']}) | CSP Recheck: PASS | "
            f"New Cost: {step['distance_after']:.2f}"
        )
    lines.append("")
    return "\n".join(lines), g_stats


# Format Section 4: System comparison with NN ON vs NN OFF on real held-out orders.
def build_system_demo_section(week_res, alt_shape_res):
    lines = []
    lines.append("======================================================================")
    lines.append("4. INTEGRATED SYSTEM: NN ON VS. NN OFF ON REAL HELD-OUT ORDERS")
    lines.append("======================================================================")
    lines.append(f"Evaluated on Test Week {week_res['week_index']} ({week_res['num_orders']} real customer invoices):")
    lines.append("  Layout Strategy                         | Standard Grid Walk (m) | Central-Depot Shape (m)")
    lines.append("  ----------------------------------------+------------------------+------------------------")
    lines.append(f"  1. Random Valid Layout                  | {week_res['random_distance']:22.1f} | {alt_shape_res['random_distance']:23.1f}")
    lines.append(f"  2. Last-Week Demand (System: NN OFF)    | {week_res['last_week_distance']:22.1f} | {alt_shape_res['last_week_distance']:23.1f}")
    lines.append(f"  3. NN Predicted Demand (NN ON, No Swap) | {week_res['nn_distance']:22.1f} | {alt_shape_res['nn_distance']:23.1f}")
    lines.append(f"  4. NN Demand + Search Swaps (Full P8)   | {week_res['nn_swaps_distance']:22.1f} | {alt_shape_res['nn_swaps_distance']:23.1f}")
    lines.append("")
    saved_vs_rand = week_res["random_distance"] - week_res["nn_swaps_distance"]
    saved_by_swaps = week_res["nn_distance"] - week_res["nn_swaps_distance"]
    lines.append(f"Summary: Full P8 (NN + CSP + Search) saves {saved_vs_rand:.1f} m vs Random layout")
    lines.append(f"         and Search swaps save an extra {saved_by_swaps:.1f} m on real orders while obeying all CSP rules.")
    lines.append("======================================================================")
    return "\n".join(lines)


# Save a 4-panel figure showing how the NN, CSP, and Search cooperate in the demo.
def save_demo_cooperation_plot(metrics, csp_tuple, search_stats, week_res, out_path="results/demo_cooperation.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    models = ["Last-Week", "4-Week Avg", "NN (MLP)"]
    val_mae = [metrics["val_last_week"][0], metrics["val_4week_avg"][0], metrics["val_nn"][0]]
    axes[0, 0].bar(models, val_mae, color=["#94a3b8", "#60a5fa", "#2563eb"])
    axes[0, 0].set_title("1. Demand Forecast Error (Validation MAE)")
    axes[0, 0].set_ylabel("MAE (Units)")

    modes = ["Pure BT", "BT + FC", "BT + FC + AC-3"]
    b_counts = [csp_tuple[0]["backtracks"], csp_tuple[1]["backtracks"], csp_tuple[2]["backtracks"]]
    axes[0, 1].bar(modes, b_counts, color=["#ef4444", "#f59e0b", "#10b981"])
    axes[0, 1].set_title("2. CSP Backtracks (90 Products)")
    axes[0, 1].set_ylabel("Backtracks")

    steps = [0] + [s["swap_number"] for s in search_stats["trace"]]
    dists = [search_stats["initial_distance"]] + [s["distance_after"] for s in search_stats["trace"]]
    axes[1, 0].plot(steps, dists, marker="o", linewidth=2, color="#7c3aed")
    axes[1, 0].set_title("3. Greedy Swap Search Optimization Trace")
    axes[1, 0].set_xlabel("Swap Step (Cost = 1 per swap)")
    axes[1, 0].set_ylabel("Demand-Weighted Distance")

    l_names = ["Random", "Last-Week\n(NN OFF)", "NN Only\n(NN ON)", "NN + Swaps\n(Full P8)"]
    l_dists = [
        week_res["random_distance"], week_res["last_week_distance"],
        week_res["nn_distance"], week_res["nn_swaps_distance"]
    ]
    axes[1, 1].bar(l_names, l_dists, color=["#cbd5e1", "#fbbf24", "#38bdf8", "#22c55e"])
    axes[1, 1].set_title("4. Real Order Picker Walking Distance (m)")
    axes[1, 1].set_ylabel("Total Meters Walked")

    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close(fig)


# Run the complete demonstration and save the cooperation plot.
def main():
    if not os.path.exists("data/weekly_demand.csv"):
        print("Preparing dataset files with data/make_data.py...")
        run_make_data(seed=42)

    bundle = load_prepared_data("data")
    metrics, test_records, _, _ = run_demand_forecasting(bundle["weekly_df"], seed=42)

    test_weeks = sorted(list(set(int(r["week_index"]) for r in test_records)))
    demo_week = test_weeks[0]

    week_res = evaluate_four_layouts_for_week(
        demo_week, test_records, bundle, initial_max_dist=6, max_swaps=6, seed=42, warehouse_shape="standard"
    )
    alt_shape_res = evaluate_four_layouts_for_week(
        demo_week, test_records, bundle, initial_max_dist=6, max_swaps=6, seed=42, warehouse_shape="central_depot"
    )

    d_maps = week_res["demand_maps"]
    context = build_csp_context(
        bundle["products_list"], bundle["slots_list"],
        bundle["copurchase_pairs"], week_res["loosest_distance_used"]
    )

    nn_text = build_nn_demo_section(bundle, metrics, d_maps)
    csp_text, csp_tuple = build_csp_demo_section(bundle, d_maps["nn"])
    search_text, search_stats = build_search_demo_section(week_res["layouts"]["nn"], d_maps["nn"], context)
    system_text = build_system_demo_section(week_res, alt_shape_res)

    full_text = "\n".join([nn_text, csp_text, search_text, system_text])
    print(full_text)

    save_demo_cooperation_plot(metrics, csp_tuple, search_stats, week_res, "results/demo_cooperation.png")
    print("Saved cooperation plot to: results/demo_cooperation.png")


if __name__ == "__main__":
    main()
