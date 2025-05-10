## Directory layout

Please check **`report.pdf`**, **`results.csv`**, and **`./problem*/p*.py`**. 

```
midterm/
├── README.md                ← this file
├── requirements.txt         ← dependencies
├── report.pdf               ← write-up
├── results.csv              ← result file
│
├── data/
│   └── OR113-2_midtermProject_data.xlsx
│
├── instances/               ← 7 × 30 NPZ test cases (Problem 4/5)
│
├── output/                  ← auxiliary dumps
│   ├── p4_results.csv
│   └── p5_results.csv
│
├── common/                  ← shared code used by P3–P5
│   ├── __init__.py
│   ├── config.py
│   ├── generate_instances.py
│   ├── experiment.py
│   ├── instance_io.py
│   ├── algo_lp.py
│   ├── algo_heuristic.py
│   └── algo_naive.py
│
├── Problem1/
│   ├── __init__.py
│   └── p1.ipynb             ← MIP with lost sales
│
├── Problem2/
│   ├── __init__.py
│   └── p2.ipynb             ← back-ordering variant
│
├── Problem3/
│   ├── __init__.py
│   └── p3.py                ← heuristic & cost report
│
├── Problem4/
│   ├── __init__.py
│   └── p4.py                ← batch run (LP + naïve)
│
└── Problem5/
    ├── __init__.py
    └── p5.py                ← batch run (LP + naïve + heuristic)
```

---

## 1  Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # pandas, numpy, gurobipy, etc.
```

---

## 2  Generate instance set (optional)

The repository already ships with the 210 `.npz` files under **`./instances`**.  
Re‑create them anytime:

```bash
python common/generate_instances.py
```

---

## 3  Re‑run experiments

### 3.1 Problem 1

open p1.ipynb in Jupyter and run all cells

### 3.1 Problem 2

open p2.ipynb in Jupyter and run all cells

### 3.3 Problem 3

```bash
python -m Problem3.p3          # writes order schedule & total cost to stdout
```

### 3.4 Problem 4 (LP + naïve)

```bash
python -m Problem4.p4
# → output/p4_results.csv
```

### 3.5 Problem 5 (LP + naïve + heuristic)

```bash
python -m Problem5.p5
# → output/p5_results.csv
```

Each script calls **`experiment.run()`** with the desired settings. Note that **`./output/p5_results.csv`** is identical to **`results.csv`**. 