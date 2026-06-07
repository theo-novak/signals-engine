# Signals Engine

Cross-sectional equity factor research toolkit with an interactive Streamlit dashboard and a CLI.
Pulls annual financial statements from SEC EDGAR, aligns them with daily price history, engineers
fundamental and momentum signals, ranks stocks into quantiles, and evaluates long-short spread
performance via rolling IC backtests.

---

## Features

- **SEC EDGAR ingestion** — XBRL company-facts API, no API key required, rate-limited automatically
- **Six fundamental signals** — earnings yield, asset growth, accruals, ROA, leverage, sales growth
- **Momentum signals** — 12-1 month (Jegadeesh & Titman 1993), 1-month reversal
- **Volatility filters** — realized vol, idiosyncratic vol, low-vol screen (Ang et al. 2006)
- **Composite signals** — z-score + linear combiner with three pre-built profiles
- **Cross-sectional ranking** — percentile ranks and N-tile buckets at each rebalance date
- **Rolling IC backtest** — Spearman IC, ICIR, hit rate, quantile spread returns, turnover
- **DuckDB storage** — local OLAP database for fundamentals and prices
- **Streamlit dashboard** — 3 interactive tabs with factor heatmaps and backtest charts
- **CLI** — `signals fetch | build | rank | backtest`

---

## Requirements

- Python 3.11 or later
- Internet access for the initial data fetch (SEC EDGAR + yfinance)

---

## Setup

```powershell
# Navigate to the project directory
cd assets/projects/signals_engine

# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# Install the package and all dependencies
pip install -e ".[dev]"
```

The `[dev]` extra installs pytest, ruff, and pytest-cov on top of the runtime dependencies.

---

## Fetching Data

Run the download script once before using the dashboard or CLI. It fetches annual 10-K
financial statements from SEC EDGAR and price history from Yahoo Finance, then persists
everything to a local DuckDB database.

```powershell
python scripts/download_data.py
```

This writes:
- `data/signals.duckdb` — fundamentals and prices tables
- `data/prices.csv` — wide (date × ticker) adjusted close CSV
- `data/cik_map.json` — ticker → CIK lookup cache (reused on subsequent runs)

The default universe is 20 large-cap US equities. Allow 5–15 minutes on first run
(SEC EDGAR enforces a 10 req/s rate limit).

### Custom universe or lookback

```powershell
# Smaller universe, 3-year price history
python scripts/download_data.py --tickers "AAPL,MSFT,GOOGL,JPM,XOM" --lookback 3

# Provide a pre-built CIK map to skip the EDGAR ticker lookup step
python scripts/download_data.py --cik-map data/cik_map.json
```

### CIK map format

```json
{
  "AAPL": "0000320193",
  "MSFT": "0000789019",
  "GOOGL": "0001652044"
}
```

Zero-padded 10-digit CIK strings. If omitted, CIKs are resolved automatically via
`https://www.sec.gov/files/company_tickers.json`.

---

## Running the Dashboard

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run src/signals_engine/app.py
```

Opens at `http://localhost:8501`. Configure the universe and parameters in the left sidebar,
then switch between tabs.

### Dashboard tabs

| Tab | What it shows |
|---|---|
| **📊 Universe & Data** | Fundamentals coverage table (period count, field completeness, date range) and normalised price history chart |
| **🔍 Factor Explorer** | Signal values with diverging colour gradient, z-scored factor exposure heatmap, cross-sectional ranking table, composite signal scores |
| **📈 Backtester** | IC / ICIR / hit rate / spread / turnover summary, cumulative spread overlay, per-signal IC bar chart and IC heatmap |

### Sidebar controls

| Control | Default | Notes |
|---|---|---|
| Universe | 20 large-caps | Comma-separated yfinance-valid tickers |
| Signal date (as-of) | Today | Date used for signal and ranking computation |
| Forward return horizon | 21 days | 21, 63, 126, or 252 trading days |
| Quantiles | 5 | Ranking buckets for long-short spread (3–10) |
| Backtest start / end | 5 years ago / today | Date range for the rolling IC backtest |

> **Note:** the dashboard requires data fetched by `download_data.py`. If the database is empty,
> Tab 1 shows a warning and Tabs 2–3 are blank. Run the fetch script first.

---

## CLI

The `signals` command is installed as a script entry point.

### Fetch data

```powershell
signals fetch --tickers "AAPL,MSFT,GOOGL,JPM" --lookback 5
```

### Print all signal values as of a date

```powershell
signals build --as-of 2024-06-30
```

### Rank the universe by a signal

```powershell
# Rank by earnings yield, show quintiles
signals rank --signal earnings_yield --quantiles 5

# Rank by ROA, show deciles
signals rank --signal return_on_assets --quantiles 10 --as-of 2024-01-31
```

### Run a rolling IC backtest

```powershell
# Earnings yield over 5 years, 21-day forward return horizon
signals backtest --signal earnings_yield --start 2019-01-01 --horizon 21

# Momentum over a specific window, quintile ranking
signals backtest --signal momentum_12_1 --start 2020-01-01 --end 2024-12-31 --quantiles 5
```

### Available signals

| Signal | Description |
|---|---|
| `earnings_yield` | EPS / Price — value signal |
| `return_on_assets` | Net income / Total assets — profitability |
| `accruals` | −(NI − OCF) / Assets — earnings quality |
| `leverage` | −Debt / Assets — balance-sheet risk |
| `asset_growth` | YoY growth in total assets |
| `sales_growth` | YoY revenue growth |
| `momentum_12_1` | 12-1 month cumulative return |

### Common options

| Flag | Default | Description |
|---|---|---|
| `--tickers` | 20 large-caps | Comma-separated ticker list |
| `--db` | `data/signals.duckdb` | DuckDB database path |
| `--as-of` | Today | Signal evaluation date (`YYYY-MM-DD`) |
| `--start` | `2019-01-01` | Backtest start date |
| `--end` | Today | Backtest end date |
| `--horizon` | 21 | Forward return horizon in trading days |
| `--quantiles` | 5 | Number of ranking buckets |

---

## Running Tests

All tests are fully offline — no network or database access required.

```powershell
.\.venv\Scripts\Activate.ps1
pytest
```

With coverage:

```powershell
pytest --cov=signals_engine --cov-report=term-missing
```

The test suite uses a deterministic seed-42 fixture (756-day synthetic prices + 5-year
fundamentals) and verifies mathematical invariants: signal sign conventions, IC bounds,
spread = long − short identity, hit rate in [0, 1], and that a predictive signal produces
positive mean IC over 36 periods.

---

## Linting

```powershell
ruff check src tests
ruff format src tests
```

---

## Project Structure

```
signals_engine/
├── src/signals_engine/
│   ├── app.py                          # Streamlit dashboard (3 tabs)
│   ├── cli.py                          # Typer CLI: fetch | build | rank | backtest
│   ├── data/
│   │   ├── schemas.py                  # Pydantic v2: Universe, FundamentalsRecord, PriceHistory
│   │   ├── edgar.py                    # SEC EDGAR XBRL fetchers (no API key required)
│   │   ├── prices.py                   # yfinance + Alpha Vantage fetchers, forward returns
│   │   └── store.py                    # DuckDB read/write for fundamentals and prices
│   ├── signals/
│   │   ├── fundamentals.py             # earnings_yield, asset_growth, accruals, ROA, leverage, sales_growth
│   │   ├── momentum.py                 # momentum_12_1, reversal_1m
│   │   ├── volatility.py               # realized_vol, idiosyncratic_vol, low_vol_filter
│   │   └── composite.py                # CompositeSignal; FUNDAMENTALS_COMPOSITE, QUALITY_VALUE, MOMENTUM_QUALITY
│   ├── rank/
│   │   ├── crosssection.py             # rank_cross_section, assign_quantiles, long_short_portfolio
│   │   └── backtest.py                 # information_coefficient, run_backtest, BacktestResult
│   └── report/
│       └── plots.py                    # Plotly: spread returns, IC bar, quantile bars, heatmaps
├── tests/
│   ├── conftest.py                     # Seed-42 price and fundamentals fixtures
│   ├── test_signals.py                 # Fundamental, momentum, and composite invariants
│   ├── test_rank.py                    # Ranking and long-short portfolio mechanics
│   └── test_backtest.py                # IC, backtest engine, rebalance date generation
├── scripts/
│   └── download_data.py                # SEC EDGAR + yfinance → DuckDB + CSV
├── notebooks/
│   └── factor_research.ipynb           # Research memo: data → signals → backtest
├── data/                               # Git-ignored; populated by download_data.py
├── dashboard.html                      # Stlite (browser-side) demo dashboard
├── .env.example                        # Configuration template
├── pyproject.toml
└── README.md
```

---

## Configuration

Copy `.env.example` to `.env` and fill in any values you need:

```ini
# Alpha Vantage API key (optional — only needed when DATA_SOURCE=alpha_vantage)
AV_API_KEY=your_key_here

# Primary price source: yfinance (default) | alpha_vantage
DATA_SOURCE=yfinance

# DuckDB file path (relative to project root)
DB_PATH=data/signals.duckdb

# Seconds between SEC EDGAR API requests (SEC fair-use guidance: ≤10 req/s)
EDGAR_SLEEP=0.12
```

The dashboard and CLI both default to `yfinance` and work without any API key.

---

## Key Concepts

### Information Coefficient (IC)

Spearman rank correlation between a signal value at rebalance date *t* and the
*h*-day forward return. IC > 0 means the signal is directionally correct on average.
ICIR = mean(IC) / std(IC) is the IC Sharpe ratio; values above 0.5 indicate a useful signal.

### Signal sign convention

All signals are signed so that **higher = more desirable**. Leverage and accruals are
negated so that low-debt, high-quality-earnings firms rank at the top.

### Composite signal construction

1. Each component signal is winsorised at ±3σ to limit outlier influence.
2. It is z-scored cross-sectionally (mean 0, std 1).
3. Z-scores are summed with user-supplied weights.

NaN values are handled per-signal, so a missing field for one company does not zero
out the composite for that company if other signals are available.

### Pre-built composites

| Name | Weights |
|---|---|
| `FUNDAMENTALS_COMPOSITE` | earnings_yield ×1, ROA ×1, accruals ×0.5, leverage ×0.5 |
| `QUALITY_VALUE` | earnings_yield ×1, ROA ×1, accruals ×1, leverage ×0.5, asset_growth ×−0.5 |
| `MOMENTUM_QUALITY` | momentum_12_1 ×1, ROA ×0.5, accruals ×0.5 |

---

## Data Sources

| Data | Source | Notes |
|---|---|---|
| Annual financial statements | SEC EDGAR XBRL (`data.sec.gov`) | Free, no API key. CIK lookup via `sec.gov/files/company_tickers.json`. |
| Adjusted daily close prices | Yahoo Finance via `yfinance` | Free, no API key. Any exchange-listed symbol. |
| Adjusted daily close prices (alt) | Alpha Vantage | Free tier: 25 req/day. Requires `AV_API_KEY`. |
