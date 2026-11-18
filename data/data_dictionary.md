# Data Dictionary – P8 Warehouse Re-Slotting from Predicted Demand

This document describes every dataset file and column used in **P8 – Warehouse Re-Slotting from Predicted Demand**, explicitly distinguishing **Real Columns** (from the UCI Online Retail II dataset) from **Generated/Simulated Columns** (created deterministically with `seed=42` in `data/make_data.py`).

---

## 1. Raw Source Dataset (`data/online_retail_II.xlsx` / `data/online_retail_II_combined.csv`)

- **Source**: UCI Machine Learning Repository – Online Retail II (ID 502)
- **Date Range**: 1 December 2009 to 9 December 2011 (`105` calendar weeks)
- **Format Adaptation**: The local dataset in `data/online_retail_II.xlsx` contains two sheets (`Year 2009-2010` with 525,461 rows and `Year 2010-2011` with 541,910 rows, totaling **1,067,371 raw rows**). `data/make_data.py` combines both sheets into `data/online_retail_II_combined.csv` locally so repeated runs do not have to re-parse the Excel file.

| Column Name | Type | Real or Generated? | Description |
| :--- | :--- | :--- | :--- |
| `Invoice` | String | **Real** | 6-digit invoice number assigned to each transaction. Invoices starting with `'C'` indicate cancellations and are removed during cleaning. |
| `StockCode` | String | **Real** | Unique product item code. Non-merchandise codes (`POST`, `DOT`, `M`, `BANK CHARGES`, `PADS`, `CRUK`, `ADJUST`, `D`) are excluded. |
| `Description` | String | **Real** | Product name/description in plain text. |
| `Quantity` | Integer | **Real** | Number of units purchased in the transaction line. Filtered to keep strictly positive values (`Quantity > 0`). |
| `InvoiceDate` | Datetime | **Real** | Timestamp when the invoice was generated. Used to group sales into sequential Monday-start weeks (`week_index`). |
| `Price` | Float | **Real** | Unit price in GBP (£). Filtered to keep strictly positive values (`Price > 0.0`). |
| `Customer ID` | Float/String | **Real** | Unique customer identifier (may be missing for guest checkouts; we keep valid guest invoices so total product volume is accurate). |
| `Country` | String | **Real** | Country of the customer (predominantly United Kingdom). |

---

## 2. Weekly Product Demand Table (`data/weekly_demand.csv`)

Built by aggregating cleaned transactions for the **top 120 products by total unit volume** across all `105` weeks (`120 × 105 = 12,600` product-week rows).

| Column Name | Type | Real or Generated? | How It Was Made / Formula |
| :--- | :--- | :--- | :--- |
| `StockCode` | String | **Real** | Product code (one of the top 120 products by total units sold). |
| `week_index` | Integer | **Real (Derived)** | Sequential week number (`0` to `104`) measured from the first Monday in December 2009. |
| `week_of_year` | Integer | **Real (Derived)** | ISO calendar week of the year (`1` to `52`), capturing seasonal patterns such as Christmas shopping spikes. |
| `units` | Float | **Real (Aggregated)** | Sum of `Quantity` sold for that `StockCode` during `week_index` (`0.0` if no units were sold that week). |
| `log_units` | Float | **Real (Derived)** | $\ln(1 + \text{units})$, used as the stabilized input lag feature for the Neural Network. |
| `mean_price` | Float | **Real (Aggregated)** | Mean unit `Price` of the product in that week (if `units == 0`, imputed with the product's historical median price). |
| `invoice_count` | Integer | **Real (Aggregated)** | Count of distinct `Invoice` numbers containing this product during `week_index`. |
| `split` | String | **Generated (Rule)** | Chronological dataset split: `history_only` (first 8 weeks used as initial lags), `train` (next 72 weeks), `val` (next 15 weeks), `test` (final 10 weeks). |

---

## 3. Co-Purchase Pairs Table (`data/copurchase_pairs.csv`)

Computed strictly from invoices in the **training weeks only** (`split == "train"` or `"history_only"`) to prevent data leakage into validation or test weeks.

| Column Name | Type | Real or Generated? | How It Was Made / Formula |
| :--- | :--- | :--- | :--- |
| `rank` | Integer | **Real (Derived)** | Rank from `1` to `30` (the 30 strongest co-bought product pairs). |
| `product_a` | String | **Real** | First `StockCode` in the co-purchased pair. |
| `product_b` | String | **Real** | Second `StockCode` in the co-purchased pair. |
| `co_purchase_invoices` | Integer | **Real (Aggregated)** | Number of distinct training-period invoices in which `product_a` and `product_b` were bought together. |

---

## 4. Product Metadata Table (`data/products.csv`)

Combines real product descriptions and volumes with simulated physical warehouse constraints generated using `random.Random(42)`.

| Column Name | Type | Real or Generated? | How It Was Made / Formula |
| :--- | :--- | :--- | :--- |
| `product_rank` | Integer | **Real (Derived)** | Volume rank from `1` (highest total units sold) to `120`. |
| `StockCode` | String | **Real** | Unique product code. |
| `Description` | String | **Real** | First non-null product description from the raw dataset. |
| `total_volume` | Integer | **Real (Aggregated)** | Total units sold across the entire cleaned dataset. |
| `is_heavy` | Integer (0/1) | **Generated (`seed=42`)** | Set to `1` with probability `0.20` using `rng.random() < 0.20`. Heavy products must be placed on low shelves (`shelf_level == 0`) in the CSP. |
| `category` | String | **Generated (`seed=42`)** | Uniformly sampled from 5 warehouse categories (`Home_Decor`, `Cleaning_Chemical`, `Food_Kitchen`, `Paper_Craft`, `Metal_Garden`). |
| `incompatible_category` | String | **Generated (Rule)** | `Cleaning_Chemical` and `Food_Kitchen` are mutually incompatible and cannot be placed in the same warehouse `aisle`. |

---

## 5. Simulated Warehouse Grid (`data/warehouse_slots.csv`)

Defines the `120` physical shelf slots (`6 aisles × 10 bays × 2 shelf levels`).

| Column Name | Type | Real or Generated? | How It Was Made / Formula |
| :--- | :--- | :--- | :--- |
| `slot_id` | String | **Generated** | Unique slot identifier from `S000` to `S119`. |
| `aisle` | Integer | **Generated** | Warehouse aisle number (`0` to `5`). Used for the aisle-incompatibility rule and grid distance calculation. |
| `bay` | Integer | **Generated** | Bay position along the aisle (`0` to `9`, where `0` is closest to the front cross-aisle). |
| `shelf_level` | Integer | **Generated** | Vertical shelf tier (`0` = low/ground shelf, `1` = upper shelf). |
| `is_low_shelf` | Integer (0/1) | **Generated** | `1` when `shelf_level == 0`, else `0` (60 low-shelf slots in total). |
| `capacity` | Integer | **Generated** | `1` (each slot holds exactly one product). |
| `distance_from_depot` | Integer | **Generated (Formula)** | Walking distance in meters from the corner depot `(0,0)`: $\text{aisle} \times 2 + \text{bay} + 1 + \text{shelf\_level}$. |
| `alt_shape_distance` | Integer | **Generated (Formula)** | Walking distance for the second warehouse shape (central depot at aisle 2.5): $\lfloor |\text{aisle} - 2.5| \times 2 + \text{bay} \times 1.5 + 1 + \text{shelf\_level} \rfloor$. |

---

## 6. Held-Out Test Orders (`data/test_orders.csv`)

Contains the actual customer orders (invoices) from the final **10 test weeks**, used by `integration/simulator.py` to measure real picker walking distance.

| Column Name | Type | Real or Generated? | How It Was Made / Formula |
| :--- | :--- | :--- | :--- |
| `week_index` | Integer | **Real (Derived)** | Test week index (`95` to `104`, representing the final 10 weeks). |
| `Invoice` | String | **Real** | Real customer invoice number. |
| `items` | String | **Real (Aggregated)** | Pipe-separated list (`|`) of distinct top-120 `StockCode` items purchased in that invoice. |
| `total_units` | Integer | **Real (Aggregated)** | Total units of top-120 items purchased in that invoice. |
