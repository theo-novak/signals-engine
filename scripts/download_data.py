"""Offline data download script.

Usage (from the project root after `pip install -e .`):

    python scripts/download_data.py
    python scripts/download_data.py --tickers AAPL,MSFT,GOOGL --lookback 7
    python scripts/download_data.py --cik-map cik_map.json

Produces:
  data/signals.duckdb     -- DuckDB with fundamentals + prices tables
  data/prices.csv         -- Wide (date × ticker) adjusted close CSV

CIK map (optional):
  A JSON file mapping uppercase ticker to zero-padded 10-digit CIK string.
  Example: {"AAPL": "0000320193", "MSFT": "0000789019"}
  If not provided, the script auto-looks up CIKs via the EDGAR company_tickers endpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download fundamentals + prices.")
    parser.add_argument("--tickers", default=(
        "AAPL,MSFT,GOOGL,AMZN,META,JPM,BAC,WFC,JNJ,UNH,"
        "XOM,CVX,PG,KO,PEP,HD,LOW,CAT,GE,MMM"
    ))
    parser.add_argument("--lookback", type=int, default=7,
                        help="Years of price history to download")
    parser.add_argument("--db", default="data/signals.duckdb")
    parser.add_argument("--cik-map", default=None,
                        help="Path to JSON file with ticker→CIK mappings")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    db_path = args.db

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    cik_lookup: dict[str, str] = {}
    if args.cik_map:
        cik_lookup = json.loads(Path(args.cik_map).read_text())

    from signals_engine.data.edgar import build_fundamentals_df, ticker_to_cik
    from signals_engine.data.prices import fetch_prices_yfinance
    from signals_engine.data.store import init_schema, write_fundamentals, write_prices
    import pandas as pd

    init_schema(db_path)

    # ── Fundamentals ────────────────────────────────────────────────────────
    print(f"\nFetching SEC EDGAR fundamentals for {len(tickers)} tickers...")
    frames = []
    for t in tickers:
        cik = cik_lookup.get(t)
        if not cik:
            print(f"  Looking up CIK for {t}...", end=" ", flush=True)
            cik = ticker_to_cik(t)
            if cik:
                cik_lookup[t] = cik
        if not cik:
            print(f"\n  WARNING: CIK not found for {t}, skipping fundamentals.")
            continue
        try:
            df = build_fundamentals_df(cik, t)
            if not df.empty:
                frames.append(df)
                print(f"  {t}: {len(df)} annual periods")
        except Exception as e:
            print(f"  {t}: ERROR — {e}")

    if frames:
        combined = pd.concat(frames).reset_index()
        n = write_fundamentals(combined, db_path=db_path)
        print(f"\nFundamentals: {n} rows written to {db_path}")

    # ── Prices ───────────────────────────────────────────────────────────────
    print(f"\nFetching prices for {len(tickers)} tickers ({args.lookback}y)...")
    ph = fetch_prices_yfinance(tickers, lookback_years=args.lookback)
    write_prices(ph.prices, db_path=db_path)
    ph.prices.to_csv("data/prices.csv")
    print(f"Prices: {ph.prices.shape[0]} days × {ph.prices.shape[1]} tickers → "
          f"{db_path} and data/prices.csv")

    # Save updated CIK map
    if cik_lookup:
        Path("data/cik_map.json").write_text(json.dumps(cik_lookup, indent=2))
        print("CIK map saved to data/cik_map.json")

    print("\nDone. Launch the app with: streamlit run src/signals_engine/app.py")


if __name__ == "__main__":
    main()
