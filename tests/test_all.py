# tests/test_all.py
# This test script runs 10 clear test cases:
#   - 5 Normal cases (known-answer search, CSP modes, NN vs baselines, simulator, integration)
#   - 3 Edge cases (single product, all-zero demand, automatic distance relaxation)
#   - 2 Impossible cases (more heavy items than low shelves, more products than slots)
# Every impossible case prints a clear "NO SOLUTION" message.

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from csp.warehouse_csp import (
    build_csp_context, solve_csp_single_distance,
    solve_csp_with_relaxation, verify_full_layout_csp
)
from search.swap_search import (
    greedy_best_first_search, bounded_ucs, astar_search,
    run_small_known_answer_example
)
from integration.simulator import simulate_single_order_distance
from integration.pipeline import load_prepared_data
from nn.demand_model import run_demand_forecasting


# Normal Test 1: Search on a small 4-product example with a known mathematical answer.
def test_normal_1_known_answer_search():
    best_layout, stats = run_small_known_answer_example()
    assert stats["initial_distance"] == 2370.0, "Expected initial distance 2370.0"
    assert stats["final_distance"] == 660.0, "Expected known optimal distance 660.0"
    assert stats["swaps_count"] == 1, "Expected exactly 1 swap"
    assert best_layout["P1"] == "S0", "High-demand P1 should move to closest slot S0"
    assert stats["nodes_expanded"] >= 1 and stats["nodes_generated"] >= 2
    print("[PASS] Normal Test 1: Small known-answer search (2370.0 -> 660.0 in 1 swap)")


# Normal Test 2: CSP solver with BT, BT+FC, and BT+FC+AC3 all produce valid layouts.
def test_normal_2_csp_modes():
    bundle = load_prepared_data("data")
    prods = bundle["products_list"][:40]
    slots = bundle["slots_list"][:50]
    pairs = bundle["copurchase_pairs"]
    demand = {str(p["StockCode"]): float(50 - i) for i, p in enumerate(prods)}

    lay_bt, st_bt = solve_csp_single_distance(prods, slots, pairs, demand, 6, use_fc=False, use_ac3=False)
    lay_fc, st_fc = solve_csp_single_distance(prods, slots, pairs, demand, 6, use_fc=True, use_ac3=False)
    lay_ac3, st_ac3 = solve_csp_single_distance(prods, slots, pairs, demand, 6, use_fc=True, use_ac3=True)

    ctx = build_csp_context(prods, slots, pairs, 6)
    assert lay_bt is not None and verify_full_layout_csp(lay_bt, ctx)
    assert lay_fc is not None and verify_full_layout_csp(lay_fc, ctx)
    assert lay_ac3 is not None and verify_full_layout_csp(lay_ac3, ctx)
    assert st_ac3["backtracks"] <= st_bt["backtracks"], "AC-3/FC should not backtrack more than pure BT"
    print(f"[PASS] Normal Test 2: CSP modes valid (BT backtracks={st_bt['backtracks']}, AC3 backtracks={st_ac3['backtracks']})")


# Normal Test 3: Neural Network and both baselines produce valid non-negative forecasts.
def test_normal_3_nn_and_baselines():
    bundle = load_prepared_data("data")
    metrics, test_records, _, _ = run_demand_forecasting(bundle["weekly_df"], seed=42)
    assert len(test_records) > 0, "Test records should not be empty"
    for key in ["val_last_week", "val_4week_avg", "val_nn", "test_last_week", "test_4week_avg", "test_nn"]:
        mae, rmse = metrics[key]
        assert mae > 0.0 and rmse >= mae, f"Invalid metrics for {key}"
    for rec in test_records[:50]:
        assert rec["pred_nn"] >= 0.0, "NN predictions must be non-negative"
    print(f"[PASS] Normal Test 3: NN vs Baselines (Val NN MAE={metrics['val_nn'][0]}, Test NN MAE={metrics['test_nn'][0]})")


# Normal Test 4: Picker walking-distance simulator on both warehouse shapes.
def test_normal_4_picker_simulator():
    slots = [
        {"slot_id": "S0", "aisle": 0, "bay": 0, "shelf_level": 0, "distance_from_depot": 1, "alt_shape_distance": 6},
        {"slot_id": "S1", "aisle": 1, "bay": 2, "shelf_level": 0, "distance_from_depot": 5, "alt_shape_distance": 7},
    ]
    slot_map = {"S0": slots[0], "S1": slots[1]}
    layout = {"A": "S0", "B": "S1"}
    dist_std = simulate_single_order_distance(["A", "B"], layout, slot_map, "standard")
    dist_alt = simulate_single_order_distance(["A", "B"], layout, slot_map, "central_depot")
    assert dist_std == 10.0, f"Expected 10.0m, got {dist_std}"
    assert dist_alt == 17.0, f"Expected 17.0m, got {dist_alt}"
    print(f"[PASS] Normal Test 4: Picker simulator verified on both shapes (std={dist_std}m, alt={dist_alt}m)")


# Normal Test 5: End-to-end CSP + Greedy, UCS, and A* swap search preserves all CSP constraints.
def test_normal_5_search_variants_preserve_csp():
    bundle = load_prepared_data("data")
    prods = bundle["products_list"][:60]
    slots = bundle["slots_list"][:60]
    pairs = bundle["copurchase_pairs"]
    demand = {str(p["StockCode"]): float((i * 37) % 300 + 10) for i, p in enumerate(prods)}

    init_layout, csp_stats = solve_csp_with_relaxation(prods, slots, pairs, demand, initial_max_distance=6)
    ctx = build_csp_context(prods, slots, pairs, csp_stats["loosest_distance_used"])

    lay_g, st_g = greedy_best_first_search(init_layout, demand, ctx, max_swaps=6)
    lay_u, st_u = bounded_ucs(init_layout, demand, ctx, max_swaps=6)
    lay_a, st_a = astar_search(init_layout, demand, ctx, max_swaps=6)

    assert verify_full_layout_csp(lay_g, ctx), "Greedy layout must satisfy CSP"
    assert verify_full_layout_csp(lay_u, ctx), "UCS layout must satisfy CSP"
    assert verify_full_layout_csp(lay_a, ctx), "A* layout must satisfy CSP"
    assert st_g["final_distance"] <= st_g["initial_distance"]
    print(f"[PASS] Normal Test 5: Greedy/UCS/A* preserve CSP rules (Greedy cost {st_g['initial_distance']} -> {st_g['final_distance']})")


# Edge Test 1: Warehouse with only 1 product and 1 slot.
def test_edge_1_single_product():
    prods = [{"StockCode": "ONLY1", "is_heavy": 1, "category": "Home_Decor", "incompatible_category": "None"}]
    slots = [{"slot_id": "S000", "aisle": 0, "bay": 0, "shelf_level": 0, "is_low_shelf": 1, "distance_from_depot": 1}]
    demand = {"ONLY1": 50.0}
    layout, _ = solve_csp_with_relaxation(prods, slots, [], demand)
    ctx = build_csp_context(prods, slots, [], 4)
    after_layout, s_stats = greedy_best_first_search(layout, demand, ctx, max_swaps=6)
    assert after_layout == {"ONLY1": "S000"} and s_stats["swaps_count"] == 0
    print("[PASS] Edge Test 1: Single-product warehouse handled cleanly with 0 swaps")


# Edge Test 2: Week where all products have 0.0 predicted demand.
def test_edge_2_all_zero_demand():
    bundle = load_prepared_data("data")
    prods = bundle["products_list"][:20]
    slots = [s for s in bundle["slots_list"] if int(s["bay"]) < 2]
    zero_demand = {str(p["StockCode"]): 0.0 for p in prods}
    layout, _ = solve_csp_with_relaxation(prods, slots, [], zero_demand)
    ctx = build_csp_context(prods, slots, [], 6)
    after_layout, s_stats = greedy_best_first_search(layout, zero_demand, ctx, max_swaps=6)
    assert layout is not None and verify_full_layout_csp(after_layout, ctx)
    assert s_stats["initial_distance"] == 0.0 and s_stats["final_distance"] == 0.0
    print("[PASS] Edge Test 2: All-zero demand week produces valid layout with 0.0 walking cost")


# Edge Test 3: Co-purchase distance rule is too tight initially and relaxes step by step.
def test_edge_3_distance_relaxation():
    prods = [
        {"StockCode": "A", "is_heavy": 1, "category": "Home_Decor", "incompatible_category": "None"},
        {"StockCode": "B", "is_heavy": 1, "category": "Home_Decor", "incompatible_category": "None"},
    ]
    slots = [
        {"slot_id": "S0", "aisle": 0, "bay": 0, "shelf_level": 0, "is_low_shelf": 1, "distance_from_depot": 1},
        {"slot_id": "S1", "aisle": 2, "bay": 0, "shelf_level": 0, "is_low_shelf": 1, "distance_from_depot": 5},
    ]
    pairs = [{"product_a": "A", "product_b": "B"}]
    demand = {"A": 100.0, "B": 90.0}
    layout, stats = solve_csp_with_relaxation(prods, slots, pairs, demand, initial_max_distance=1, step=2)
    assert layout is not None, "Relaxation should find a valid assignment once distance reaches 5"
    assert stats["loosest_distance_used"] == 5 and stats["relaxed_steps"] == 2
    print(f"[PASS] Edge Test 3: Distance relaxation succeeded (relaxed 1 -> 3 -> {stats['loosest_distance_used']})")


# Impossible Test 1: More heavy products (3) than available low shelves (1).
def test_impossible_1_too_many_heavy_items():
    print("Running Impossible Test 1 (expecting clear NO SOLUTION message below):")
    prods = [
        {"StockCode": "H1", "is_heavy": 1, "category": "Home_Decor"},
        {"StockCode": "H2", "is_heavy": 1, "category": "Home_Decor"},
        {"StockCode": "H3", "is_heavy": 1, "category": "Home_Decor"},
    ]
    slots = [
        {"slot_id": "S0", "aisle": 0, "bay": 0, "shelf_level": 0, "is_low_shelf": 1, "distance_from_depot": 1},
        {"slot_id": "S1", "aisle": 0, "bay": 0, "shelf_level": 1, "is_low_shelf": 0, "distance_from_depot": 2},
        {"slot_id": "S2", "aisle": 0, "bay": 1, "shelf_level": 1, "is_low_shelf": 0, "distance_from_depot": 3},
    ]
    demand = {"H1": 10.0, "H2": 10.0, "H3": 10.0}
    layout, _ = solve_csp_with_relaxation(prods, slots, [], demand)
    assert layout is None, "Must return None when heavy items exceed low shelves"
    print("[PASS] Impossible Test 1: Correctly rejected impossible heavy-shelf count")


# Impossible Test 2: More products (4) than total warehouse slots (2).
def test_impossible_2_more_products_than_slots():
    print("Running Impossible Test 2 (expecting clear NO SOLUTION message below):")
    prods = [
        {"StockCode": "P1", "is_heavy": 0, "category": "Home_Decor"},
        {"StockCode": "P2", "is_heavy": 0, "category": "Home_Decor"},
        {"StockCode": "P3", "is_heavy": 0, "category": "Home_Decor"},
        {"StockCode": "P4", "is_heavy": 0, "category": "Home_Decor"},
    ]
    slots = [
        {"slot_id": "S0", "aisle": 0, "bay": 0, "shelf_level": 0, "is_low_shelf": 1, "distance_from_depot": 1},
        {"slot_id": "S1", "aisle": 0, "bay": 1, "shelf_level": 0, "is_low_shelf": 1, "distance_from_depot": 2},
    ]
    demand = {"P1": 10.0, "P2": 10.0, "P3": 10.0, "P4": 10.0}
    layout, _ = solve_csp_with_relaxation(prods, slots, [], demand)
    assert layout is None, "Must return None when products exceed slots"
    print("[PASS] Impossible Test 2: Correctly rejected more products than slots")


# Run all 10 tests in order and print a summary.
def run_all_tests():
    print("======================================================================")
    print("RUNNING P8 AUTOMATED TEST SUITE (5 NORMAL, 3 EDGE, 2 IMPOSSIBLE CASES)")
    print("======================================================================")
    test_normal_1_known_answer_search()
    test_normal_2_csp_modes()
    test_normal_3_nn_and_baselines()
    test_normal_4_picker_simulator()
    test_normal_5_search_variants_preserve_csp()
    test_edge_1_single_product()
    test_edge_2_all_zero_demand()
    test_edge_3_distance_relaxation()
    test_impossible_1_too_many_heavy_items()
    test_impossible_2_more_products_than_slots()
    print("======================================================================")
    print("ALL 10 TESTS PASSED SUCCESSFULLY!")
    print("======================================================================")


if __name__ == "__main__":
    run_all_tests()
