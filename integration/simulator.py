# integration/simulator.py
# This module simulates a warehouse worker (picker) walking through the warehouse
# to collect real customer orders (invoices) from the 10 held-out test weeks.
# Supports both the Standard Warehouse Shape and a Second Warehouse Shape (optional extension).

from csp.warehouse_csp import slot_to_slot_distance


# Return the distance from the packing depot to a slot for either warehouse shape.
# Shape 1 ('standard') has the depot at corner (0, 0).
# Shape 2 ('central_depot') has the depot in the middle of the front cross-aisle.
def get_depot_distance(slot, warehouse_shape="standard"):
    if warehouse_shape == "central_depot":
        return float(slot.get("alt_shape_distance", slot["distance_from_depot"]))
    return float(slot["distance_from_depot"])


# Sort a list of slots in a simple S-traversal order (by aisle, then by bay)
# so the picker walks through aisles in a logical path instead of zig-zagging randomly.
def order_slots_for_picking_tour(slots_to_visit):
    scored = []
    for s in slots_to_visit:
        aisle = int(s["aisle"])
        bay = int(s["bay"])
        sid = str(s["slot_id"])
        scored.append((aisle, bay, sid, s))
    scored.sort()

    ordered = []
    for item in scored:
        ordered.append(item[3])
    return ordered


# Simulate the walking distance for a single customer invoice (list of StockCodes).
# The picker starts at the depot, visits the shelf slot of each ordered item,
# and walks back to the depot to pack the box.
def simulate_single_order_distance(product_codes, layout, slot_map, warehouse_shape="standard"):
    visited_slots = []
    seen_slot_ids = set()

    for code in product_codes:
        if code in layout:
            sid = layout[code]
            if sid not in seen_slot_ids and sid in slot_map:
                seen_slot_ids.add(sid)
                visited_slots.append(slot_map[sid])

    if len(visited_slots) == 0:
        return 0.0

    ordered_tour = order_slots_for_picking_tour(visited_slots)
    # Walk from depot to the first item's slot
    total_walk = get_depot_distance(ordered_tour[0], warehouse_shape)

    # Walk from each item's slot to the next item's slot in the order
    for i in range(len(ordered_tour) - 1):
        step_dist = slot_to_slot_distance(ordered_tour[i], ordered_tour[i + 1])
        total_walk += float(step_dist)

    # Walk from the final item's slot back to the packing depot
    total_walk += get_depot_distance(ordered_tour[-1], warehouse_shape)
    return total_walk


# Simulate total picker walking distance across all real customer invoices in one week.
def simulate_week_orders(week_orders_df, layout, slots_list, warehouse_shape="standard"):
    slot_map = {}
    for s in slots_list:
        slot_map[str(s["slot_id"])] = s

    total_distance = 0.0
    order_count = 0

    for row in week_orders_df.itertuples(index=False):
        items_str = str(row.items)
        if len(items_str) == 0:
            continue
        product_codes = items_str.split("|")
        dist = simulate_single_order_distance(product_codes, layout, slot_map, warehouse_shape)
        total_distance += dist
        order_count += 1

    return round(total_distance, 2), order_count
