# search/swap_search.py
# This module improves an initial warehouse layout by swapping pairs of products.
# Every candidate swap is rechecked by the CSP to guarantee all rules stay satisfied.
# Implements Greedy Best-First Search (main), Bounded UCS (baseline), and an A* variant.

from csp.warehouse_csp import verify_full_layout_csp, slot_to_slot_distance


# Calculate the demand-weighted walking distance of a layout.
# High-demand products placed far from the depot create a large walking cost,
# and co-bought pairs placed closer together also reduce walking effort.
def calculate_walking_distance(layout, demand_map, context):
    slot_map = context["slot_map"]
    total_cost = 0.0

    for product_code, slot_id in layout.items():
        units = float(demand_map.get(product_code, 0.0))
        dist = float(slot_map[slot_id]["distance_from_depot"])
        # Round-trip from depot to slot is 2 * distance
        total_cost += units * 2.0 * dist

    # Add a small co-purchase walking term for the top 30 co-bought pairs
    for pa, neighbors in context["co_neighbors"].items():
        if pa in layout:
            slot_a = slot_map[layout[pa]]
            for pb in neighbors:
                if pb in layout and pa < pb:
                    slot_b = slot_map[layout[pb]]
                    pair_dist = slot_to_slot_distance(slot_a, slot_b)
                    total_cost += 15.0 * pair_dist

    return round(total_cost, 2)


# Return a new layout dictionary where product_a and product_b have traded slots.
def swap_two_products(layout, product_a, product_b):
    new_layout = dict(layout)
    slot_a = new_layout[product_a]
    slot_b = new_layout[product_b]
    new_layout[product_a] = slot_b
    new_layout[product_b] = slot_a
    return new_layout


# Select the candidate products allowed to participate in swaps:
# the 30 highest-demand products plus products sitting in the 30 slots nearest the depot.
def get_swap_candidates(layout, demand_map, context, limit=30):
    scored_prods = []
    for code in layout.keys():
        dem = float(demand_map.get(code, 0.0))
        scored_prods.append((dem, code))
    scored_prods.sort(reverse=True)

    top_demand_codes = []
    for item in scored_prods[:limit]:
        top_demand_codes.append(item[1])

    # Also include products currently located in the closest slots near the depot
    slot_map = context["slot_map"]
    scored_by_slot = []
    for code, sid in layout.items():
        d = int(slot_map[sid]["distance_from_depot"])
        scored_by_slot.append((d, code))
    scored_by_slot.sort()

    near_depot_codes = []
    for item in scored_by_slot[:limit]:
        near_depot_codes.append(item[1])

    combined = []
    for code in top_demand_codes + near_depot_codes:
        if code not in combined:
            combined.append(code)
    return combined


# Generate valid improving neighbor states by swapping pairs of candidate products
# and rechecking every swap with the CSP verifier.
def generate_valid_swap_neighbors(layout, current_dist, candidates, demand_map, context, max_neighbors=12):
    neighbors = []
    n = len(candidates)

    for i in range(n):
        pa = candidates[i]
        for j in range(i + 1, n):
            pb = candidates[j]
            swapped = swap_two_products(layout, pa, pb)
            new_dist = calculate_walking_distance(swapped, demand_map, context)

            # Only keep swaps that strictly reduce walking distance and pass CSP rules
            if new_dist < current_dist - 0.01:
                if verify_full_layout_csp(swapped, context):
                    neighbors.append((new_dist, pa, pb, swapped))

    neighbors.sort()
    return neighbors[:max_neighbors]


# Convert a layout dictionary into a sorted tuple so we can remember visited states.
def make_state_signature(layout):
    pairs = []
    for k in sorted(layout.keys()):
        pairs.append((k, layout[k]))
    return tuple(pairs)


# Compute priority score f(n) for the chosen search algorithm:
# - 'greedy': h(n) = current walking distance (main algorithm)
# - 'ucs':    g(n) = number of swaps so far (cost = 1 per swap, baseline)
# - 'astar':  g(n) + h(n), where h(n) estimates remaining swaps needed (not guaranteed optimal)
def compute_node_priority(algorithm, g_swaps, current_dist, target_dist):
    if algorithm == "ucs":
        return float(g_swaps)
    if algorithm == "astar":
        # Estimate remaining swaps by dividing remaining distance gap by 1500.0 units.
        # Because a single swap of a top seller can save more than 1500.0 units,
        # h(n) can overestimate true swaps left, so this A* variant is not guaranteed optimal.
        remaining_gap = max(0.0, current_dist - target_dist)
        h_est = remaining_gap / 1500.0
        return float(g_swaps) + h_est
    # Default is Greedy Best-First Search: priority is purely the heuristic distance
    return float(current_dist)


# Core swap search engine supporting 'greedy', 'ucs', and 'astar'.
# Goal: walking distance <= target_dist OR 6 swaps reached.
def run_swap_search(initial_layout, demand_map, context, algorithm="greedy",
                    max_swaps=6, target_reduction_ratio=0.022, max_expansions=35):
    init_dist = calculate_walking_distance(initial_layout, demand_map, context)
    target_dist = round(init_dist * (1.0 - target_reduction_ratio), 2)
    candidates = get_swap_candidates(initial_layout, demand_map, context, limit=30)

    start_prio = compute_node_priority(algorithm, 0, init_dist, target_dist)
    # Each frontier item is (priority, g_swaps, current_dist, node_id, layout, trace_list)
    # The integer node_id breaks ties so Python never compares two layout dictionaries.
    frontier = [(start_prio, 0, init_dist, 0, dict(initial_layout), [])]
    visited = set([make_state_signature(initial_layout)])

    nodes_expanded = 0
    nodes_generated = 1
    best_dist = init_dist
    best_layout = dict(initial_layout)
    best_trace = []

    while len(frontier) > 0 and nodes_expanded < max_expansions:
        frontier.sort()
        prio, g_swaps, cur_dist, _, cur_layout, cur_trace = frontier.pop(0)
        nodes_expanded += 1

        if cur_dist < best_dist:
            best_dist = cur_dist
            best_layout = cur_layout
            best_trace = cur_trace

        # Goal test: reached target walking distance or hit the 6-swap limit
        if cur_dist <= target_dist or g_swaps >= max_swaps:
            if algorithm != "ucs":
                break
            continue

        neighbors = generate_valid_swap_neighbors(
            cur_layout, cur_dist, candidates, demand_map, context, max_neighbors=8
        )
        for new_dist, pa, pb, next_layout in neighbors:
            sig = make_state_signature(next_layout)
            if sig in visited:
                continue
            visited.add(sig)
            next_g = g_swaps + 1
            next_prio = compute_node_priority(algorithm, next_g, new_dist, target_dist)
            step_info = {
                "swap_number": next_g,
                "product_a": pa,
                "product_b": pb,
                "slot_a": cur_layout[pa],
                "slot_b": cur_layout[pb],
                "distance_after": new_dist
            }
            frontier.append((
                next_prio, next_g, new_dist, nodes_generated,
                next_layout, cur_trace + [step_info]
            ))
            nodes_generated += 1

    return best_layout, {
        "algorithm": algorithm,
        "initial_distance": init_dist,
        "final_distance": best_dist,
        "target_distance": target_dist,
        "swaps_count": len(best_trace),
        "nodes_expanded": nodes_expanded,
        "nodes_generated": nodes_generated,
        "trace": best_trace
    }


# Convenience wrapper for the main algorithm: Greedy Best-First Search.
def greedy_best_first_search(initial_layout, demand_map, context, max_swaps=6):
    return run_swap_search(initial_layout, demand_map, context, algorithm="greedy", max_swaps=max_swaps)


# Convenience wrapper for the baseline: Bounded Uniform Cost Search.
def bounded_ucs(initial_layout, demand_map, context, max_swaps=6, max_expansions=35):
    return run_swap_search(initial_layout, demand_map, context, algorithm="ucs",
                           max_swaps=max_swaps, max_expansions=max_expansions)


# Convenience wrapper for the A* Search variant.
def astar_search(initial_layout, demand_map, context, max_swaps=6):
    return run_swap_search(initial_layout, demand_map, context, algorithm="astar", max_swaps=max_swaps)


# Small 4-product, 4-slot toy example with a mathematically known optimal answer
# (Deliverable 5: search tested on a small example with a known answer).
def run_small_known_answer_example():
    from csp.warehouse_csp import build_csp_context
    products = [
        {"StockCode": "P1", "is_heavy": 0, "category": "Home_Decor", "incompatible_category": "None"},
        {"StockCode": "P2", "is_heavy": 0, "category": "Home_Decor", "incompatible_category": "None"},
        {"StockCode": "P3", "is_heavy": 0, "category": "Home_Decor", "incompatible_category": "None"},
        {"StockCode": "P4", "is_heavy": 0, "category": "Home_Decor", "incompatible_category": "None"},
    ]
    slots = [
        {"slot_id": "S0", "aisle": 0, "bay": 0, "shelf_level": 0, "is_low_shelf": 1, "distance_from_depot": 1},
        {"slot_id": "S1", "aisle": 0, "bay": 1, "shelf_level": 0, "is_low_shelf": 1, "distance_from_depot": 2},
        {"slot_id": "S2", "aisle": 1, "bay": 1, "shelf_level": 0, "is_low_shelf": 1, "distance_from_depot": 5},
        {"slot_id": "S3", "aisle": 2, "bay": 2, "shelf_level": 0, "is_low_shelf": 1, "distance_from_depot": 10},
    ]
    demand = {"P1": 100.0, "P2": 40.0, "P3": 20.0, "P4": 5.0}
    initial_layout = {"P4": "S0", "P2": "S1", "P3": "S2", "P1": "S3"}
    context = build_csp_context(products, slots, [], max_distance=10)

    best_layout, stats = greedy_best_first_search(initial_layout, demand, context, max_swaps=1)
    return best_layout, stats
