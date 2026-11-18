# integration/pipeline.py
# This module connects the three AI parts into one pipeline:
# 1. Neural Network predicts product demand for the upcoming week.
# 2. CSP builds a valid initial layout (high demand near the depot, obeying all rules).
# 3. Search tries pairwise swaps among the top 30 products, rechecking every swap with CSP.
# 4. Simulator tests the final layouts on real held-out customer invoices.

import os
import random
import pandas as pd
from nn.demand_model import run_demand_forecasting
from csp.warehouse_csp import build_csp_context, solve_csp_with_relaxation, verify_full_layout_csp
from search.swap_search import greedy_best_first_search, bounded_ucs, astar_search
from integration.simulator import simulate_week_orders


# Load all preprocessed CSV tables from the data/ folder into memory.
def load_prepared_data(data_dir="data"):
    weekly_df = pd.read_csv(os.path.join(data_dir, "weekly_demand.csv"), dtype={"StockCode": str})
    products_df = pd.read_csv(os.path.join(data_dir, "products.csv"), dtype={"StockCode": str})
    slots_df = pd.read_csv(os.path.join(data_dir, "warehouse_slots.csv"), dtype={"slot_id": str})
    copurchase_df = pd.read_csv(
        os.path.join(data_dir, "copurchase_pairs.csv"),
        dtype={"product_a": str, "product_b": str}
    )
    test_orders_df = pd.read_csv(os.path.join(data_dir, "test_orders.csv"), dtype={"Invoice": str})

    return {
        "weekly_df": weekly_df,
        "products_list": products_df.to_dict(orient="records"),
        "slots_list": slots_df.to_dict(orient="records"),
        "copurchase_pairs": copurchase_df.to_dict(orient="records"),
        "test_orders_df": test_orders_df
    }


# Build demand dictionaries ({StockCode: units}) for a specific test week
# from the test records produced by the demand forecasting module.
def get_week_demand_maps(test_records, week_index, products_list, seed=42):
    rng = random.Random(seed + int(week_index))
    random_map = {}
    for p in products_list:
        random_map[str(p["StockCode"])] = rng.uniform(1.0, 100.0)

    last_week_map = {}
    four_week_map = {}
    nn_map = {}
    actual_map = {}

    for rec in test_records:
        if int(rec["week_index"]) == int(week_index):
            code = str(rec["StockCode"])
            last_week_map[code] = float(rec["pred_last_week"])
            four_week_map[code] = float(rec["pred_4week_avg"])
            nn_map[code] = float(rec["pred_nn"])
            actual_map[code] = float(rec["target_units"])

    return {
        "random": random_map,
        "last_week": last_week_map,
        "four_week": four_week_map,
        "nn": nn_map,
        "actual": actual_map
    }


# Run the full pipeline for one week:
# build the 4 layouts (Random, Last-Week, NN, NN + Swaps) and test them on real orders.
def evaluate_four_layouts_for_week(week_index, test_records, data_bundle,
                                   initial_max_dist=6, max_swaps=6, seed=42,
                                   warehouse_shape="standard"):
    products_list = data_bundle["products_list"]
    slots_list = data_bundle["slots_list"]
    copurchase_pairs = data_bundle["copurchase_pairs"]
    test_orders_df = data_bundle["test_orders_df"]
    week_orders = test_orders_df[test_orders_df["week_index"] == int(week_index)]

    d_maps = get_week_demand_maps(test_records, week_index, products_list, seed=seed)

    # 1. Random valid CSP layout
    layout_rand, csp_rand = solve_csp_with_relaxation(
        products_list, slots_list, copurchase_pairs, d_maps["random"],
        initial_max_distance=initial_max_dist
    )
    # 2. Last-week demand CSP layout (system with NN OFF)
    layout_lw, csp_lw = solve_csp_with_relaxation(
        products_list, slots_list, copurchase_pairs, d_maps["last_week"],
        initial_max_distance=initial_max_dist
    )
    # 3. NN predicted demand CSP layout (system with NN ON before swaps)
    layout_nn, csp_nn = solve_csp_with_relaxation(
        products_list, slots_list, copurchase_pairs, d_maps["nn"],
        initial_max_distance=initial_max_dist
    )
    # 4. NN demand + Greedy Best-First Swap Search (rechecking every swap with CSP)
    dist_used = csp_nn["loosest_distance_used"]
    context = build_csp_context(products_list, slots_list, copurchase_pairs, dist_used)
    layout_nn_swaps, search_stats = greedy_best_first_search(
        layout_nn, d_maps["nn"], context, max_swaps=max_swaps
    )

    dist_rand, n_orders = simulate_week_orders(week_orders, layout_rand, slots_list, warehouse_shape)
    dist_lw, _ = simulate_week_orders(week_orders, layout_lw, slots_list, warehouse_shape)
    dist_nn, _ = simulate_week_orders(week_orders, layout_nn, slots_list, warehouse_shape)
    dist_nn_swaps, _ = simulate_week_orders(week_orders, layout_nn_swaps, slots_list, warehouse_shape)

    return {
        "week_index": int(week_index),
        "num_orders": n_orders,
        "random_distance": dist_rand,
        "last_week_distance": dist_lw,
        "nn_distance": dist_nn,
        "nn_swaps_distance": dist_nn_swaps,
        "loosest_distance_used": dist_used,
        "search_stats": search_stats,
        "csp_stats": csp_nn,
        "layouts": {
            "random": layout_rand,
            "last_week": layout_lw,
            "nn": layout_nn,
            "nn_swaps": layout_nn_swaps
        },
        "demand_maps": d_maps
    }
