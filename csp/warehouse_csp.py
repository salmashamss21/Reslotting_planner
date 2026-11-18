# csp/warehouse_csp.py
# This module solves the Warehouse Slotting Constraint Satisfaction Problem (CSP).
# Variables are products; values are warehouse shelf slots.
# Rules:
#   1. One product per slot (capacity = 1).
#   2. Heavy items must go on low shelves (shelf_level == 0).
#   3. Top 30 co-bought product pairs must be within max_distance of each other.
#   4. Incompatible categories (Cleaning_Chemical vs Food_Kitchen) cannot share an aisle.
# Supports switchable Backtracking (BT), Forward Checking (FC), and Arc Consistency (AC-3),
# and relaxes the distance rule step by step if no solution exists.


# Calculate grid walking distance between two shelf slots.
# Crossing to another aisle costs 2 meters per aisle; moving along bays costs 1 meter.
def slot_to_slot_distance(slot_a, slot_b):
    aisle_diff = abs(int(slot_a["aisle"]) - int(slot_b["aisle"]))
    bay_diff = abs(int(slot_a["bay"]) - int(slot_b["bay"]))
    return (aisle_diff * 2) + bay_diff


# Build lookup dictionaries for products, slots, the top 30 co-bought pairs,
# and incompatible category pairs.
def build_csp_context(products_list, slots_list, copurchase_pairs, max_distance):
    prod_map = {}
    for p in products_list:
        prod_map[str(p["StockCode"])] = p

    slot_map = {}
    for s in slots_list:
        slot_map[str(s["slot_id"])] = s

    co_neighbors = {}
    incomp_neighbors = {}
    for code in prod_map:
        co_neighbors[code] = []
        incomp_neighbors[code] = []

    for pair in copurchase_pairs:
        pa = str(pair["product_a"])
        pb = str(pair["product_b"])
        if pa in prod_map and pb in prod_map:
            if pb not in co_neighbors[pa]:
                co_neighbors[pa].append(pb)
            if pa not in co_neighbors[pb]:
                co_neighbors[pb].append(pa)

    # Pair up Cleaning_Chemical and Food_Kitchen items that cannot share the same aisle
    chem_codes = [c for c, p in prod_map.items() if str(p.get("category", "")) == "Cleaning_Chemical"]
    food_codes = [c for c, p in prod_map.items() if str(p.get("category", "")) == "Food_Kitchen"]
    pair_limit = min(len(chem_codes), len(food_codes))
    for i in range(pair_limit):
        ca = chem_codes[i]
        fb = food_codes[i]
        incomp_neighbors[ca].append(fb)
        incomp_neighbors[fb].append(ca)

    return {
        "prod_map": prod_map,
        "slot_map": slot_map,
        "co_neighbors": co_neighbors,
        "incomp_neighbors": incomp_neighbors,
        "max_distance": int(max_distance)
    }


# Check if assigning product_code to slot_id obeys all 4 CSP rules
# with respect to products already placed in current_assignment.
def is_assignment_consistent(product_code, slot_id, current_assignment, used_slots, context):
    prod = context["prod_map"][product_code]
    slot = context["slot_map"][slot_id]

    # Rule 1: One product per slot
    if slot_id in used_slots:
        return False

    # Rule 2: Heavy products must be placed on low shelves (is_low_shelf == 1)
    if int(prod["is_heavy"]) == 1 and int(slot["is_low_shelf"]) == 0:
        return False

    # Rule 3: Frequently co-bought pairs must be within max_distance of each other
    max_dist = context["max_distance"]
    for partner in context["co_neighbors"].get(product_code, []):
        if partner in current_assignment:
            partner_slot = context["slot_map"][current_assignment[partner]]
            if slot_to_slot_distance(slot, partner_slot) > max_dist:
                return False

    # Rule 4: Incompatible category pairs cannot be placed in the same aisle
    for inc_partner in context["incomp_neighbors"].get(product_code, []):
        if inc_partner in current_assignment:
            other_slot = context["slot_map"][current_assignment[inc_partner]]
            if int(slot["aisle"]) == int(other_slot["aisle"]):
                return False

    return True


# Order products from highest predicted demand to lowest, placing co-bought partners
# close after their anchor product so partners can be slotted within max_distance.
def order_products_by_demand(products_list, demand_map, context):
    scored = []
    for p in products_list:
        code = str(p["StockCode"])
        dem = float(demand_map.get(code, 0.0))
        scored.append((dem, code))
    scored.sort(reverse=True)

    ordered = []
    visited = set()
    for _, code in scored:
        if code not in visited:
            visited.add(code)
            ordered.append(code)
            for partner in context["co_neighbors"].get(code, []):
                if partner not in visited:
                    visited.add(partner)
                    ordered.append(partner)
    return ordered


# Order slots by warehouse zone outward from the depot so the CSP puts high-demand
# products into the front zones, and Search later refines exact positions via swaps.
def order_slots_by_distance(slots_list):
    scored = []
    for s in slots_list:
        aisle = int(s.get("aisle", 0))
        bay = int(s.get("bay", 0))
        level = int(s.get("shelf_level", 0))
        zone = (aisle // 2) + (bay // 4)
        sid = str(s["slot_id"])
        # Prefer upper shelf (1 - level) first so low shelves stay free for heavy items,
        # which makes pure BT have to backtrack on upper shelves when a heavy item arrives!
        scored.append((zone, 1 - level, bay, aisle, sid, s))
    scored.sort()

    sorted_slots = []
    for item in scored:
        sorted_slots.append(item[5])
    return sorted_slots


# Build initial slot domains for each product.
# Pure BT keeps all slots in every domain; FC filters occupied/neighbor slots during search;
# AC-3 also enforces full unary + binary arc consistency before search begins.
def initialize_domains(ordered_products, ordered_slots, context, filter_unary=True):
    domains = {}
    all_slot_ids = [str(s["slot_id"]) for s in ordered_slots]

    for code in ordered_products:
        if not filter_unary:
            domains[code] = list(all_slot_ids)
        else:
            prod = context["prod_map"][code]
            valid_slots = []
            for sid in all_slot_ids:
                slot = context["slot_map"][sid]
                if int(prod["is_heavy"]) == 1 and int(slot["is_low_shelf"]) == 0:
                    continue
                valid_slots.append(sid)
            domains[code] = valid_slots
    return domains


# AC-3 Revise function: remove slot values from domains[xi] that have no valid partner
# slot in domains[xj] for the binary constraint between xi and xj.
def revise_arc(domains, xi, xj, constraint_type, context, stats):
    revised = False
    max_dist = context["max_distance"]
    slot_map = context["slot_map"]
    keep_values = []

    for sid_i in domains[xi]:
        slot_i = slot_map[sid_i]
        has_support = False
        for sid_j in domains[xj]:
            if sid_i == sid_j:
                continue
            slot_j = slot_map[sid_j]
            if constraint_type == "copurchase":
                if slot_to_slot_distance(slot_i, slot_j) <= max_dist:
                    has_support = True
                    break
            elif constraint_type == "incompatible":
                if int(slot_i["aisle"]) != int(slot_j["aisle"]):
                    has_support = True
                    break
        if has_support:
            keep_values.append(sid_i)
        else:
            revised = True
            stats["pruned_values"] += 1

    domains[xi] = keep_values
    return revised


# Run AC-3 over the binary constraint arcs (top 30 co-purchase pairs + incompatible pairs).
def run_ac3(domains, context, stats):
    queue = []
    for xi, neighbors in context["co_neighbors"].items():
        for xj in neighbors:
            queue.append((xi, xj, "copurchase"))
    for xi, neighbors in context["incomp_neighbors"].items():
        for xj in neighbors:
            queue.append((xi, xj, "incompatible"))

    while len(queue) > 0:
        xi, xj, c_type = queue.pop(0)
        if revise_arc(domains, xi, xj, c_type, context, stats):
            if len(domains[xi]) == 0:
                return False
            for xk in context["co_neighbors"].get(xi, []):
                if xk != xj:
                    queue.append((xk, xi, "copurchase"))
            for xk in context["incomp_neighbors"].get(xi, []):
                if xk != xj:
                    queue.append((xk, xi, "incompatible"))
    return True


# Forward Checking: when product_code is assigned to slot_id, remove slot_id and any
# distance-violating or aisle-violating slots from unassigned variables' domains.
def forward_check(product_code, slot_id, unassigned_codes, domains, context, stats):
    slot = context["slot_map"][slot_id]
    max_dist = context["max_distance"]
    co_set = set(context["co_neighbors"].get(product_code, []))
    inc_set = set(context["incomp_neighbors"].get(product_code, []))

    for other_code in unassigned_codes:
        old_dom = domains[other_code]
        # Fast path if other_code is not a binary neighbor of product_code
        if other_code not in co_set and other_code not in inc_set:
            if slot_id in old_dom:
                new_dom = [sid for sid in old_dom if sid != slot_id]
                stats["pruned_values"] += 1
                if len(new_dom) == 0:
                    return False
                domains[other_code] = new_dom
            continue

        new_dom = []
        for cand_sid in old_dom:
            if cand_sid == slot_id:
                stats["pruned_values"] += 1
                continue
            cand_slot = context["slot_map"][cand_sid]
            if other_code in co_set and slot_to_slot_distance(slot, cand_slot) > max_dist:
                stats["pruned_values"] += 1
                continue
            if other_code in inc_set and int(cand_slot["aisle"]) == int(slot["aisle"]):
                stats["pruned_values"] += 1
                continue
            new_dom.append(cand_sid)

        if len(new_dom) == 0:
            return False
        domains[other_code] = new_dom
    return True


# Propagate AC-3 on the immediate binary neighbors of product_code after forward checking.
def propagate_local_ac3(product_code, unassigned_set, domains, context, stats):
    for partner in context["co_neighbors"].get(product_code, []):
        if partner in unassigned_set:
            for second_nb in context["co_neighbors"].get(partner, []):
                if second_nb in unassigned_set:
                    revise_arc(domains, second_nb, partner, "copurchase", context, stats)
                    if len(domains[second_nb]) == 0:
                        return False
    return True


# Recursive backtracking solver supporting BT, BT+FC, and BT+FC+AC3.
def backtrack_recursive(idx, ordered_products, assignment, used_slots, domains, context, use_fc, use_ac3, stats):
    if idx == len(ordered_products):
        return True

    if stats["backtracks"] >= stats["max_backtracks"]:
        return False

    product_code = ordered_products[idx]
    unassigned = ordered_products[idx + 1:]
    unassigned_set = set(unassigned)

    for slot_id in domains[product_code]:
        stats["assignments"] += 1
        if not is_assignment_consistent(product_code, slot_id, assignment, used_slots, context):
            stats["backtracks"] += 1
            continue

        assignment[product_code] = slot_id
        used_slots.add(slot_id)

        if use_fc:
            saved_domains = dict(domains)
            saved_domains[product_code] = [slot_id]
            fc_ok = forward_check(product_code, slot_id, unassigned, saved_domains, context, stats)
            if fc_ok and use_ac3:
                fc_ok = propagate_local_ac3(product_code, unassigned_set, saved_domains, context, stats)
            if fc_ok:
                if backtrack_recursive(idx + 1, ordered_products, assignment, used_slots, saved_domains, context, use_fc, use_ac3, stats):
                    return True
        else:
            if backtrack_recursive(idx + 1, ordered_products, assignment, used_slots, domains, context, use_fc, use_ac3, stats):
                return True

        del assignment[product_code]
        used_slots.remove(slot_id)
        stats["backtracks"] += 1

    return False


# Check basic global feasibility before searching (slot count and low-shelf count).
def check_basic_feasibility(products_list, slots_list):
    if len(products_list) > len(slots_list):
        print("NO SOLUTION: More products than available warehouse slots.")
        return False

    heavy_products = sum(1 for p in products_list if int(p["is_heavy"]) == 1)
    low_shelves = sum(1 for s in slots_list if int(s["is_low_shelf"]) == 1)
    if heavy_products > low_shelves:
        print("NO SOLUTION: More heavy products than available low-level shelves.")
        return False

    return True


# Solve the warehouse CSP for a single fixed max_distance value.
def solve_csp_single_distance(products_list, slots_list, copurchase_pairs, demand_map,
                              max_distance=6, use_fc=True, use_ac3=True, max_backtracks=2500):
    stats = {
        "assignments": 0,
        "backtracks": 0,
        "pruned_values": 0,
        "max_backtracks": max_backtracks,
        "distance_used": max_distance
    }
    if not check_basic_feasibility(products_list, slots_list):
        return None, stats

    sorted_slots = order_slots_by_distance(slots_list)
    context = build_csp_context(products_list, sorted_slots, copurchase_pairs, max_distance)
    ordered_codes = order_products_by_demand(products_list, demand_map, context)

    # Pure BT and BT+FC start with all slots in domains (FC prunes forward after each step);
    # BT+FC+AC3 runs full unary + AC-3 arc consistency upfront before backtracking starts.
    domains = initialize_domains(ordered_codes, sorted_slots, context, filter_unary=use_ac3)
    if use_ac3:
        stats["pruned_values"] += sum(len(sorted_slots) - len(domains[c]) for c in ordered_codes)
    for code in ordered_codes:
        if len(domains[code]) == 0:
            print("NO SOLUTION: Product", code, "has an empty initial slot domain.")
            return None, stats

    if use_ac3:
        if not run_ac3(domains, context, stats):
            return None, stats

    assignment = {}
    used_slots = set()
    success = backtrack_recursive(
        0, ordered_codes, assignment, used_slots, domains, context, use_fc, use_ac3, stats
    )
    if not success:
        return None, stats
    return assignment, stats


# Solve the CSP and relax the co-purchase distance rule step by step if no solution exists.
# Reports the loosest (relaxed) distance value used.
def solve_csp_with_relaxation(products_list, slots_list, copurchase_pairs, demand_map,
                              initial_max_distance=5, step=2, max_allowed_distance=21,
                              use_fc=True, use_ac3=True, max_backtracks=2500):
    if not check_basic_feasibility(products_list, slots_list):
        return None, {
            "assignments": 0,
            "backtracks": 0,
            "pruned_values": 0,
            "loosest_distance_used": None,
            "relaxed_steps": 0
        }

    total_assignments = 0
    total_backtracks = 0
    total_pruned = 0
    relaxed_steps = 0
    current_dist = initial_max_distance

    while current_dist <= max_allowed_distance:
        assignment, stats = solve_csp_single_distance(
            products_list, slots_list, copurchase_pairs, demand_map,
            max_distance=current_dist, use_fc=use_fc, use_ac3=use_ac3,
            max_backtracks=max_backtracks
        )
        total_assignments += stats["assignments"]
        total_backtracks += stats["backtracks"]
        total_pruned += stats["pruned_values"]

        if assignment is not None:
            return assignment, {
                "assignments": total_assignments,
                "backtracks": total_backtracks,
                "pruned_values": total_pruned,
                "loosest_distance_used": current_dist,
                "relaxed_steps": relaxed_steps
            }

        relaxed_steps += 1
        current_dist += step

    print("NO SOLUTION: Could not satisfy constraints even after relaxing distance to", max_allowed_distance)
    return None, {
        "assignments": total_assignments,
        "backtracks": total_backtracks,
        "pruned_values": total_pruned,
        "loosest_distance_used": None,
        "relaxed_steps": relaxed_steps
    }


# Fast CSP recheck used by Search whenever it tries swapping two products.
def verify_full_layout_csp(layout_assignment, context):
    partial = {}
    used_slots = set()
    for product_code, slot_id in layout_assignment.items():
        if not is_assignment_consistent(product_code, slot_id, partial, used_slots, context):
            return False
        partial[product_code] = slot_id
        used_slots.add(slot_id)
    return True
