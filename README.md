# PEI Weather Station Analysis and Fire Weather Index (FWI) Calculator

This project helps you:

1) Clean and standardize weather station CSV data
2) Compare stations to see whether some are collecting overlapping information (redundancy)
3) Calculate daily Fire Weather Index (FWI) values to support wildfire-risk monitoring

It is written to be runnable by non–data science users.

## Quick Start 

Clone this repository onto your local computer. 

Place the raw HOBOlink data files into `data/raw/`

In VSCode open 'cleaning.py' and run it.

Note the first time you run cleaning.py it will take quite some time. Subsequent runs will take much less time. 

Then open `analysis.ipynb` in VS Code and click **Run All**.

## What’s In This Repo

- `src/cleaning.py`: Cleans raw data, standardizes timestamps to UTC, and produces a single hourly dataset.
- `analysis.ipynb`: Generates charts and tables, runs redundancy analysis, and calculates/validates FWI.
- `requirements.txt`: Python packages used by the project.

Key folders (already part of the repo structure):

- `data/raw/`: Raw station data inputs (HOBOlink station exports) and cached ECCC downloads
- `data/scrubbed/`: Clean outputs (hourly combined dataset + QA summaries)
- `outputs/`: Figures, tables, and logs created by the scripts/notebook

## Prerequisites

- Python 3.10+ recommended
- An internet connection (used by the cleaning step to download/cache ECCC reference data if not already present)

To run the notebook, you need either:

- **VS Code** with the **Python** and **Jupyter** extensions, or
- The `jupyter` Python package installed (see Setup below)

## Setup

1. **Create a Python environment (recommended)**

   ```bash
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install packages**

   ```bash
   pip install -r requirements.txt
   ```

3. **Install Jupyter (needed to run `analysis.ipynb`)**

   ```bash
   pip install jupyter
   ```

## Data Requirements (Important)

- Place station CSV exports under `data/raw/` using the existing folder structure.
- Do not rename the station folders unless you also update the code.
- The cleaning script also caches ECCC reference data under:
  - `data/raw/ECCC_Stanhope/`
  - `data/raw/ECCC_Stanhope_FWI/`

If you received this repository with `data/raw/` already populated, you can usually skip manual data placement.

## Run Instructions

### Step 1 — Clean and Prepare the Data

Run:

```bash
python src/cleaning.py
```

What this does (high level):

- Reads station data from `data/raw/`
- Downloads/caches ECCC reference data if needed
- Standardizes timestamps to UTC and resamples to hourly data
- Writes cleaned outputs to `data/scrubbed/` and logs to `outputs/logs/`

What “success” looks like:

- `data/scrubbed/02_hourly_weather_utc.csv`
- `data/scrubbed/02_missingness_hourly_summary.csv`
- `data/scrubbed/02_qc_out_of_range_counts.csv`
- `outputs/logs/` contains a new log file for the run

### Step 2 — Run the Analysis and FWI Calculation

Option A (recommended): **VS Code**

1. Open `analysis.ipynb` in VS Code
2. Choose the Python interpreter from your virtual environment (the `venv` you created)
3. Click **Run All**

Option B: **Command line (browser notebook)**

```bash
jupyter notebook analysis.ipynb
```

In the browser window, use **Run → Run All Cells**, then scroll to see results.

## Troubleshooting

- **“python is not recognized” (Windows):** Install Python from python.org and re-open PowerShell.
- **Virtual environment won’t activate (PowerShell):** You may need to allow local scripts:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- **Notebook won’t run / kernel not found:** Ensure you installed `jupyter` and that VS Code is using the `venv` interpreter.
- **Downloads fail / run is slow:** Check your internet connection; ECCC data downloads are cached under `data/raw/` for re-use.
