# P8 – Warehouse Re-Slotting from Predicted Demand
**Coursework Project: Neural Network Demand Forecasting + CSP Slot Assignment + Heuristic Swap Search**

This project solves the weekly **Warehouse Re-Slotting** problem on the **UCI Online Retail II** dataset (`1,067,371` transactions). Each week, the system:
1. **Forecasts** next-week units sold per product using a 2-layer Multi-Layer Perceptron (`nn/demand_model.py`) and compares it against **Last-Week** and **4-Week Moving Average** baselines.
2. **Constructs** a valid initial shelf layout using a switchable **Constraint Satisfaction Problem (CSP)** solver (`csp/warehouse_csp.py` supporting **Backtracking**, **Forward Checking**, and **AC-3**), placing high-demand items near the packing depot while obeying capacity, ground-level heavy shelf, co-purchase distance, and aisle-incompatibility rules (with automatic step-by-step distance relaxation when needed).
3. **Refines** the layout using **Heuristic Swap Search** (`search/swap_search.py` supporting **Greedy Best-First Search**, **Bounded UCS**, and an **A\* Variant**) over pairwise swaps among the 30 highest-demand products and near-depot slots, rechecking every swap with the CSP verifier.
4. **Simulates** a human warehouse picker walking through aisles to fulfill **4,318 real held-out customer invoices** over 10 test weeks (`integration/simulator.py`).

---

## 1. Environment Setup & Pinned Requirements

Only 5 external Python libraries are required, all pinned in [requirements.txt](file:///d:/Projects/Intelligent%20Systems/requirements.txt):

```powershell
# 1. Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install the pinned dependencies
.\.venv\Scripts\pip.exe install -r requirements.txt
```

---

## 2. Dataset Placement & Format Note

Because the UCI Online Retail II file is ~45 MB, raw and generated data files inside `data/` (`*.xlsx` and `*.csv`) are listed in `.gitignore`.
- **Where to put the dataset**: Place the UCI Online Retail II file inside the `data/` folder as `data/online_retail_II.xlsx` (or `data/online_retail_II_combined.csv`).
- **Dataset Format Adaptation**: While some descriptions refer to Online Retail II as a single CSV, the official UCI distribution (`online_retail_II.xlsx`) is an Excel workbook split across two sheets (`Year 2009-2010` with 525,461 rows and `Year 2010-2011` with 541,910 rows, totaling **1,067,371 rows**) with column names `['Invoice', 'StockCode', 'Description', 'Quantity', 'InvoiceDate', 'Price', 'Customer ID', 'Country']`. Our [make_data.py](file:///d:/Projects/Intelligent%20Systems/data/make_data.py) script automatically reads both sheets, caches `data/online_retail_II_combined.csv` locally for fast re-runs, cleans cancellations (`Invoice` starting with `'C'`), selects the top 120 products by volume, and outputs the processed CSV tables.

---

## 3. Exact Run Commands

Run all commands from the project root directory:

```powershell
# Step 1: Preprocess the local dataset and generate warehouse tables (seed=42)
.\.venv\Scripts\python.exe data/make_data.py

# Step 2: Run the 10-case automated test suite (5 normal, 3 edge, 2 impossible)
.\.venv\Scripts\python.exe tests/test_all.py

# Step 3: Run the 3 multi-seed experiments (5 seeds: 42..46) and save tables/plots to results/
.\.venv\Scripts\python.exe eval/run_experiments.py

# Step 4: Run the 5-minute live demo (prints full trace and saves results/demo_cooperation.png)
.\.venv\Scripts\python.exe run_demo.py
```

---

## 4. Project Structure

```text
├── data/
│   ├── __init__.py
│   ├── make_data.py                 # Local dataset cleaner, top-120 selector, co-purchase & grid builder
│   └── data_dictionary.md           # Documentation of all real vs. simulated columns and formulas
├── nn/
│   ├── __init__.py
│   └── demand_model.py              # MLPRegressor + Last-Week and 4-Week-Average baselines
├── csp/
│   ├── __init__.py
│   └── warehouse_csp.py             # Switchable BT, FC, AC-3 solver + distance rule relaxation
├── search/
│   ├── __init__.py
│   └── swap_search.py               # Greedy Best-First, Bounded UCS, A* variant + known-answer example
├── integration/
│   ├── __init__.py
│   ├── simulator.py                 # Picker walking-distance simulator on real invoices (2 shapes)
│   └── pipeline.py                  # End-to-end NN -> CSP -> Swap Search (with CSP recheck) pipeline
├── eval/
│   ├── __init__.py
│   └── run_experiments.py           # Multi-seed experiments (5 seeds) generating tables & PNG plots
├── tests/
│   ├── __init__.py
│   └── test_all.py                  # 5 normal, 3 edge, and 2 impossible test cases
├── results/                         # Generated CSV tables, Markdown summaries, and PNG plots
├── report/
│   ├── report.md                    # Complete project report (max 12 pages)
│   ├── presentation_outline.md      # 10-minute presentation slide outline
│   ├── demo_script.md               # 5-minute live demo script
│   └── contribution_statement_template.md # Individual contribution template + git log instructions
├── run_demo.py                      # Single-command integrated demo script
├── requirements.txt                 # Pinned Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Setup, commands, assumptions, results summary, and glossary
```

---

## 5. Short Results Summary (5 Seeds: `42, 43, 44, 45, 46`)

1. **Demand Forecasting (`nn_vs_baselines.csv`)**:
   - **Validation MAE**: Neural Network (`132.49 ± 0.74`) beats 4-Week Average (`146.89`) and Last-Week (`176.24`).
   - **Test MAE**: Neural Network (`245.28 ± 3.75`) beats 4-Week Average (`250.96`) and Last-Week (`283.73`).
2. **Experiment 1 – Swap Search Algorithms (`exp1_search_comparison.csv`)**:
   - **Greedy Best-First (Main)**: Saves **`9,514.5 ± 943.1`** walking cost in `4.4 ± 1.1` swaps, expanding only `5.4 ± 1.1` nodes (`183.8 ms`).
   - **Bounded UCS (Baseline)**: Expands `35.0` nodes breadth-first by swap count, saving `6,579.5 ± 1,160.4` in `2.0` swaps.
   - **A\* Variant**: Saves `8,574.3 ± 1,662.3` in fewer swaps (`3.4 ± 0.5` swaps), expanding `25.6 ± 13.3` nodes.
3. **Experiment 2 – CSP Solver Scaling (`exp2_csp_scaling.csv`)**:
   - At `90` products, **Pure BT** requires `4,005.0` backtracks (`0` pruned), **BT + FC** drops backtracks to `71.6 ± 25.9` (`5,771.8` pruned), and **BT + FC + AC-3** solves the layout in **0.0 backtracks** (`6,191.6` pruned).
4. **Experiment 3 – Four Layouts Over 10 Real Test Weeks (`exp3_four_layouts_10weeks.csv`, `4,318` real invoices)**:
   - **Random Valid Layout**: `166,658.6 ± 786.3 m`
   - **Last-Week Demand (NN OFF)**: `142,308.0 ± 0.0 m`
   - **NN Demand (NN ON)**: `139,350.2 ± 1,003.7 m`
   - **NN + Swaps (Full P8)**: **`138,085.2 ± 1,113.3 m`** (saving **28.6 km** vs. Random and **4.2 km** vs. Last-Week).

---

## 6. Assumptions & Design Decisions

Where the project specification left room for parameter choices, we chose the simplest, most transparent option:
1. **Warehouse Grid Geometry**: For 120 products, we use a 120-slot grid (`6 aisles × 10 bays × 2 shelf levels`, where `level 0` is the low ground shelf and `level 1` is the upper shelf). Crossing an aisle costs `2m` and moving one bay costs `1m`.
2. **Heavy & Incompatible Product Generation (`seed=42`)**: Each product is marked heavy (`is_heavy = 1`) with 20% probability (`24` heavy products placed into `60` available low shelves). Products are assigned to 5 simulated categories, with `Cleaning_Chemical` and `Food_Kitchen` paired as incompatible in the same aisle.
3. **Chronological Split Boundaries**: Across the 105 calendar weeks of Online Retail II, the first 8 weeks form the initial lag window, leaving 97 usable weeks split chronologically into **72 training weeks**, **15 validation weeks**, and **10 test weeks**.
4. **Distance Rule Relaxation Step**: When solving with relaxation (`solve_csp_with_relaxation`), if the initial `max_distance` is infeasible, it is increased in steps of `+2` meters up to `21` meters and the loosest distance used is reported.

---

## 7. Beginner Glossary (Everyday Supermarket Analogies)

- **Neural Network**: A computer model made of simple connected math units that learns patterns from historical data. *Analogy*: Like a seasoned supermarket manager who looks at the past 8 weeks of sales and the time of year to guess how many cartons of milk will sell next week.
- **Training / Validation / Test Split**: Dividing historical data by date into three separate chunks: **Training** (to teach the model), **Validation** (to tune settings), and **Test** (the final 10 weeks held back to grade the system fairly). *Analogy*: Studying homework problems (Train), taking a practice quiz (Validation), and sitting the real final exam (Test).
- **Baseline**: A simple rule-of-thumb prediction used as a reference point to see if a complex model is actually better. *Analogy*: Guessing next week's sales will simply equal last week's sales (**Last-Week Baseline**).
- **Overfitting**: When a model memorizes random one-time spikes in the training data instead of learning general patterns, causing it to make poor predictions on new future weeks. *Analogy*: Memorizing the exact answers to a practice test word-for-word instead of understanding the subject.
- **Seed (Random Seed)**: A fixed starting number (`seed=42`) given to Python's random generator so every run produces the exact same sequence of numbers. *Analogy*: Entering the same "map seed" in a game so the exact same world generates every time.
- **CSP (Constraint Satisfaction Problem)**: A puzzle where we assign a choice to every item while obeying strict rules that cannot be broken. *Analogy*: Assigning groceries to supermarket shelves where heavy bags must stay on the bottom shelf and bleach cannot sit in the food aisle.
- **Variable (in CSP)**: Each item that needs a decision made for it. Here, each of the **120 products** is a variable because we must choose which shelf slot it gets.
- **Domain (in CSP)**: The list of allowed choices remaining for a variable. Here, a product's domain is the list of **shelf slots (`S000`–`S119`)** where that product can still legally be placed.
- **Constraint (in CSP)**: A mandatory rule that a layout must obey—for example, *"only one product per slot"* or *"heavy items must go on low shelves."*
- **Backtracking (BT)**: A trial-and-error solver that places products one at a time and backs up to undo its last placement whenever it hits a broken rule. *Analogy*: Walking through a maze and stepping backward to the previous corner whenever you hit a dead end.
- **Forward Checking (FC)**: Looking one step ahead right after placing a product and immediately crossing off newly forbidden shelf slots from the remaining unplaced products' lists. *Analogy*: As soon as you put bleach in Aisle 1, crossing "Aisle 1" off the allowed list for all food items right away.
- **AC-3 (Arc Consistency 3)**: A look-ahead algorithm that checks pairs of connected variables ("arcs") in a chain and removes any shelf slot from Product A's domain if Product B has no legal partner slot left to match it. *Analogy*: Realizing before you even start placing items that if Pasta and Sauce must sit within 6 meters, and Sauce can only fit in Aisle 0, then Pasta cannot go in Aisle 5.
- **Search (State-Space Search)**: Starting from a valid shelf layout and exploring sequences of actions (**swapping two products' shelves**) to find a layout where pickers walk less.
- **Heuristic ($h(n)$)**: A fast rule-of-thumb score that estimates how good a state is. Here, our heuristic is the **demand-weighted walking distance** of the layout. *Analogy*: Judging how close you are to the bottom of a hill by checking your current elevation.
- **Greedy Best-First Search**: A search algorithm that always expands whichever neighbor layout has the lowest heuristic walking distance right now. *Analogy*: Always taking the steepest downhill step at every crossroads.
- **UCS (Uniform Cost Search)**: A search algorithm that ignores the heuristic and explores layouts strictly in order of how many swaps ($g(n)$) have been made (`1 swap`, then `2 swaps`, etc.). *Analogy*: Checking every possible 1-swap change before checking any 2-swap change.
- **A\* Search**: A search algorithm that combines the cost spent so far ($g(n)$, number of swaps) plus the heuristic estimate of remaining effort ($h(n)$) as $f(n) = g(n) + h(n)$, balancing fewer swaps against lower walking distance.
#   R e s l o t t i n g - p l a n n e r  
 