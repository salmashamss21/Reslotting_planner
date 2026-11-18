# data/make_data.py
# This script reads the local UCI Online Retail II dataset from data/,
# cleans cancellations and invalid rows, selects the top 120 products by volume,
# builds the weekly product demand table, finds the top 30 co-bought pairs
# from training weeks only, and generates the simulated warehouse grid.

import os
import math
import random
import numpy as np
import pandas as pd


# Read the local dataset from data/ without downloading anything from the internet.
# Because the raw file in data/ is an Excel workbook with two sheets, we load both
# sheets once and save a combined CSV cache so future runs finish in seconds.
def load_local_retail_data(data_dir="data"):
    cached_csv = os.path.join(data_dir, "online_retail_II_combined.csv")
    excel_path = os.path.join(data_dir, "online_retail_II.xlsx")

    if os.path.exists(cached_csv):
        print("Loading cached local CSV:", cached_csv)
        df = pd.read_csv(cached_csv, dtype={"Invoice": str, "StockCode": str})
    elif os.path.exists(excel_path):
        print("Reading local Excel file (both sheets):", excel_path)
        xl = pd.ExcelFile(excel_path)
        sheet1 = xl.parse("Year 2009-2010")
        sheet2 = xl.parse("Year 2010-2011")
        df = pd.concat([sheet1, sheet2], ignore_index=True)
        df.to_csv(cached_csv, index=False)
        print("Saved combined local CSV cache to:", cached_csv)
    else:
        raise FileNotFoundError("Could not find online_retail_II.xlsx or CSV in data/")

    print("Raw dataset row count:", len(df))
    print("Raw dataset columns:", list(df.columns))
    print("Sample raw rows:")
    print(df.head(3))
    return df


# Remove cancelled invoices (starting with 'C'), missing dates/codes,
# non-positive quantities or prices, and postage/accounting codes.
def clean_transactions(df):
    df = df.copy()
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip().str.upper()

    # Keep only rows that are NOT cancellations (do not start with C)
    not_cancelled = ~df["Invoice"].str.startswith("C")
    df = df[not_cancelled]

    # Remove missing StockCode or InvoiceDate, and require Quantity > 0 and Price > 0
    df = df.dropna(subset=["StockCode", "InvoiceDate", "Quantity", "Price"])
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0.0)]

    # Exclude non-product fee codes like postage and bank charges
    bad_codes = [
        "POST", "D", "M", "BANK CHARGES", "PADS", "DOT",
        "CRUK", "ADJUST", "ADJUST2", "TEST001", "TEST002"
    ]
    df = df[~df["StockCode"].isin(bad_codes)]
    return df


# Convert InvoiceDate into a sequential week number (0, 1, 2, ...) and week of year (1..52)
# so we can group sales chronologically week by week.
def add_week_columns(df):
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    # Floor each timestamp to the Monday of its week
    monday_dates = df["InvoiceDate"].dt.to_period("W-SUN").dt.start_time
    min_monday = monday_dates.min()
    df["week_index"] = ((monday_dates - min_monday).dt.days // 7).astype(int)
    df["week_of_year"] = df["InvoiceDate"].dt.isocalendar().week.astype(int)
    return df


# Find the top 120 products by total quantity sold across the cleaned dataset.
def select_top_products(df, num_products=120):
    totals = df.groupby("StockCode")["Quantity"].sum().reset_index()
    totals = totals.sort_values(by="Quantity", ascending=False)
    top_codes = list(totals["StockCode"].iloc[:num_products])
    return top_codes


# Build a complete table with one row per (product, week_index).
# Missing product-weeks are filled with 0 units sold so time series are continuous.
def build_weekly_table(df_top, top_codes):
    max_week = int(df_top["week_index"].max())
    # Map each week_index to its calendar week_of_year (1..52)
    week_to_woy = df_top.groupby("week_index")["week_of_year"].first().to_dict()
    default_price = df_top.groupby("StockCode")["Price"].median().to_dict()

    grouped = df_top.groupby(["StockCode", "week_index"]).agg(
        units=("Quantity", "sum"),
        mean_price=("Price", "mean"),
        invoice_count=("Invoice", "nunique")
    ).reset_index()

    lookup = {}
    for row in grouped.itertuples(index=False):
        lookup[(row.StockCode, int(row.week_index))] = (
            float(row.units), float(row.mean_price), int(row.invoice_count)
        )

    rows = []
    for code in top_codes:
        for w in range(max_week + 1):
            woy = int(week_to_woy.get(w, (w % 52) + 1))
            if (code, w) in lookup:
                units, m_price, inv_cnt = lookup[(code, w)]
            else:
                units = 0.0
                m_price = float(default_price.get(code, 1.0))
                inv_cnt = 0
            rows.append({
                "StockCode": code,
                "week_index": w,
                "week_of_year": woy,
                "units": units,
                "log_units": round(math.log(1.0 + units), 6),
                "mean_price": round(m_price, 4),
                "invoice_count": inv_cnt
            })
    return pd.DataFrame(rows)


# Label weeks chronologically: first 8 weeks are history only,
# last 10 weeks are test, 15 weeks before test are validation, rest are train (~72 weeks).
def assign_chronological_splits(weekly_df):
    weekly_df = weekly_df.copy()
    max_week = int(weekly_df["week_index"].max())
    test_start = max_week - 10 + 1
    val_start = test_start - 15

    splits = []
    for w in weekly_df["week_index"]:
        if w < 8:
            splits.append("history_only")
        elif w < val_start:
            splits.append("train")
        elif w < test_start:
            splits.append("val")
        else:
            splits.append("test")
    weekly_df["split"] = splits
    return weekly_df, val_start - 1, test_start


# Helper function to return the count of a co-purchase pair for sorting.
def get_pair_count(pair_item):
    return pair_item[1]


# Compute the 30 strongest co-purchased product pairs using ONLY training weeks.
# This avoids leaking future validation or test co-purchase patterns into the CSP.
def build_copurchase_pairs(df_top, train_max_week, top_k=30):
    train_df = df_top[df_top["week_index"] <= train_max_week]
    invoice_groups = train_df.groupby("Invoice")["StockCode"].unique()

    pair_counts = {}
    for items in invoice_groups:
        sorted_items = sorted(list(items))
        n = len(sorted_items)
        for i in range(n):
            for j in range(i + 1, n):
                pair = (sorted_items[i], sorted_items[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    sorted_pairs = sorted(pair_counts.items(), key=get_pair_count, reverse=True)
    rows = []
    for rank, (pair, count) in enumerate(sorted_pairs[:top_k]):
        rows.append({
            "rank": rank + 1,
            "product_a": pair[0],
            "product_b": pair[1],
            "co_purchase_invoices": count
        })
    return pd.DataFrame(rows)


# Generate simulated product attributes (heavy flag, category, incompatible category)
# with a fixed random seed so every run creates the exact same product table.
def generate_product_metadata(df_top, top_codes, seed=42):
    rng = random.Random(seed)
    desc_map = df_top.groupby("StockCode")["Description"].first().to_dict()
    vol_map = df_top.groupby("StockCode")["Quantity"].sum().to_dict()

    categories = ["Home_Decor", "Cleaning_Chemical", "Food_Kitchen", "Paper_Craft", "Metal_Garden"]
    rows = []
    for idx, code in enumerate(top_codes):
        # About 20% of products are marked heavy and must go on low shelves (level 0)
        is_heavy = 1 if rng.random() < 0.20 else 0
        cat_idx = rng.randint(0, len(categories) - 1)
        cat_name = categories[cat_idx]
        # Cleaning_Chemical and Food_Kitchen are incompatible in the same aisle
        incomp_cat = "Food_Kitchen" if cat_name == "Cleaning_Chemical" else (
            "Cleaning_Chemical" if cat_name == "Food_Kitchen" else "None"
        )
        rows.append({
            "product_rank": idx + 1,
            "StockCode": code,
            "Description": str(desc_map.get(code, "Product " + code))[:50],
            "total_volume": int(vol_map.get(code, 0)),
            "is_heavy": is_heavy,
            "category": cat_name,
            "incompatible_category": incomp_cat
        })
    return pd.DataFrame(rows)


# Generate 120 warehouse slots (6 aisles x 10 bays x 2 shelf levels).
# Also computes walking distance for a second warehouse shape (optional extension).
def generate_warehouse_slots():
    rows = []
    slot_num = 0
    for aisle in range(6):
        for bay in range(10):
            for level in range(2):
                slot_id = f"S{slot_num:03d}"
                # Shape 1 (Standard corner depot at (0,0)): distance grows with aisle and bay
                dist_standard = (aisle * 2) + bay + 1 + level
                # Shape 2 (Central depot at aisle 2.5, bay 0): shorter cross-aisle distance
                dist_wide = int(abs(aisle - 2.5) * 2 + (bay * 1.5) + 1 + level)
                rows.append({
                    "slot_id": slot_id,
                    "aisle": aisle,
                    "bay": bay,
                    "shelf_level": level,
                    "is_low_shelf": 1 if level == 0 else 0,
                    "capacity": 1,
                    "distance_from_depot": dist_standard,
                    "alt_shape_distance": dist_wide
                })
                slot_num += 1
    return pd.DataFrame(rows)


# Save the real customer invoices from the 10 test weeks so we can simulate
# a warehouse picker walking to fulfill actual customer orders.
def extract_test_orders(df_top, test_start_week):
    test_df = df_top[df_top["week_index"] >= test_start_week]
    order_items = {}
    order_units = {}
    order_week = {}

    for row in test_df.itertuples(index=False):
        inv = str(row.Invoice)
        code = str(row.StockCode)
        qty = int(row.Quantity)
        wk = int(row.week_index)
        if inv not in order_items:
            order_items[inv] = set()
            order_units[inv] = 0
            order_week[inv] = wk
        order_items[inv].add(code)
        order_units[inv] += qty

    rows = []
    for inv in sorted(order_items.keys()):
        sorted_codes = sorted(list(order_items[inv]))
        rows.append({
            "week_index": order_week[inv],
            "Invoice": inv,
            "items": "|".join(sorted_codes),
            "total_units": order_units[inv]
        })
    return pd.DataFrame(rows)


# Main entry point to run the full preprocessing and generation pipeline.
def main(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    data_dir = "data"

    raw_df = load_local_retail_data(data_dir)
    clean_df = clean_transactions(raw_df)
    clean_df = add_week_columns(clean_df)
    print("Cleaned transaction count:", len(clean_df))

    top_codes = select_top_products(clean_df, num_products=120)
    df_top = clean_df[clean_df["StockCode"].isin(top_codes)].copy()
    print("Transactions for top 120 products:", len(df_top))

    weekly_df = build_weekly_table(df_top, top_codes)
    weekly_df, train_max_week, test_start_week = assign_chronological_splits(weekly_df)

    copurchase_df = build_copurchase_pairs(df_top, train_max_week, top_k=30)
    products_df = generate_product_metadata(df_top, top_codes, seed=seed)
    slots_df = generate_warehouse_slots()
    test_orders_df = extract_test_orders(df_top, test_start_week)

    weekly_df.to_csv(os.path.join(data_dir, "weekly_demand.csv"), index=False)
    copurchase_df.to_csv(os.path.join(data_dir, "copurchase_pairs.csv"), index=False)
    products_df.to_csv(os.path.join(data_dir, "products.csv"), index=False)
    slots_df.to_csv(os.path.join(data_dir, "warehouse_slots.csv"), index=False)
    test_orders_df.to_csv(os.path.join(data_dir, "test_orders.csv"), index=False)

    print("Saved weekly_demand.csv, copurchase_pairs.csv, products.csv, warehouse_slots.csv, test_orders.csv")


if __name__ == "__main__":
    main(seed=42)
