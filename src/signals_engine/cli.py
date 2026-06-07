"""Typer CLI for the signals engine.

Commands:
  signals fetch   -- pull SEC EDGAR fundamentals + prices for a universe
  signals build   -- compute all signals for the universe as of a date
  signals rank    -- rank universe by a signal and show decile breakdown
  signals backtest -- run a single-signal IC backtest over a date range
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="signals", help="Cross-sectional signals engine CLI.")
console = Console()

_DEFAULT_TICKERS = "AAPL,MSFT,GOOGL,AMZN,META,JPM,BAC,WFC,JNJ,UNH,XOM,CVX,PG,KO,PEP,HD,LOW,CAT,GE,MMM"


def _parse_tickers(s: str) -> list[str]:
    return [t.strip().upper() for t in s.split(",") if t.strip()]


def _load_fund(db_path: str, tickers: list[str]) -> pd.DataFrame:
    from signals_engine.data.store import read_fundamentals
    fund = read_fundamentals(tickers, db_path=db_path)
    if fund.empty:
        console.print("[red]No fundamentals found. Run `signals fetch` first.[/red]")
        raise typer.Exit(1)
    return fund


def _load_prices_wide(db_path: str, tickers: list[str]) -> pd.DataFrame:
    from signals_engine.data.store import read_prices
    prices = read_prices(tickers, db_path=db_path)
    if prices.empty:
        console.print("[red]No prices found. Run `signals fetch` first.[/red]")
        raise typer.Exit(1)
    return prices


@app.command("fetch")
def cmd_fetch(
    tickers: str = typer.Option(_DEFAULT_TICKERS, "--tickers", "-t",
                                 help="Comma-separated ticker list"),
    lookback: int = typer.Option(5, "--lookback", "-l", help="Price lookback in years"),
    db_path: str = typer.Option("data/signals.duckdb", "--db", help="DuckDB path"),
    cik_map: Optional[Path] = typer.Option(None, "--cik-map",
                                            help="JSON file mapping ticker→CIK"),
) -> None:
    """Fetch SEC EDGAR fundamentals and yfinance prices; persist to DuckDB."""
    from signals_engine.data.edgar import build_fundamentals_df, ticker_to_cik
    from signals_engine.data.prices import fetch_prices_yfinance
    from signals_engine.data.store import init_schema, write_fundamentals, write_prices

    tick_list = _parse_tickers(tickers)
    init_schema(db_path)

    cik_lookup: dict[str, str] = {}
    if cik_map and cik_map.exists():
        cik_lookup = json.loads(cik_map.read_text())

    console.print(f"[bold]Fetching fundamentals for {len(tick_list)} tickers...[/bold]")
    fund_rows = []
    for t in tick_list:
        cik = cik_lookup.get(t.upper())
        if not cik:
            console.print(f"  [yellow]Looking up CIK for {t}...[/yellow]")
            cik = ticker_to_cik(t)
        if not cik:
            console.print(f"  [red]CIK not found for {t}, skipping.[/red]")
            continue
        try:
            df = build_fundamentals_df(cik, t)
            if not df.empty:
                fund_rows.append(df)
                console.print(f"  [green]{t}[/green]: {len(df)} annual periods")
        except Exception as exc:
            console.print(f"  [red]{t} failed: {exc}[/red]")

    if fund_rows:
        combined = pd.concat(fund_rows).reset_index()
        n = write_fundamentals(combined, db_path=db_path)
        console.print(f"[bold green]Fundamentals: {n} rows written.[/bold green]")

    console.print(f"\n[bold]Fetching prices for {len(tick_list)} tickers...[/bold]")
    ph = fetch_prices_yfinance(tick_list, lookback_years=lookback)
    n2 = write_prices(ph.prices, db_path=db_path)
    console.print(f"[bold green]Prices: {n2} rows written.[/bold green]")


@app.command("build")
def cmd_build(
    tickers: str = typer.Option(_DEFAULT_TICKERS, "--tickers", "-t"),
    as_of: Optional[str] = typer.Option(None, "--as-of",
                                         help="Signal date YYYY-MM-DD (default: today)"),
    db_path: str = typer.Option("data/signals.duckdb", "--db"),
) -> None:
    """Compute all signals for the universe as of a date and print a summary table."""
    from signals_engine.signals.fundamentals import all_fundamental_signals
    from signals_engine.signals.momentum import all_momentum_signals
    from signals_engine.rank.crosssection import rank_universe

    tick_list = _parse_tickers(tickers)
    d = datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else date.today()
    fund = _load_fund(db_path, tick_list)
    prices = _load_prices_wide(db_path, tick_list)

    fund_sigs = all_fundamental_signals(fund, prices, d)
    mom_sigs = all_momentum_signals(prices, d)
    all_sigs = pd.concat([fund_sigs, mom_sigs], axis=1).dropna(how="all")

    table = Table(title=f"Signals as of {d}", show_lines=True)
    table.add_column("Ticker", style="cyan")
    for col in all_sigs.columns:
        table.add_column(col, justify="right")

    for ticker, row in all_sigs.iterrows():
        table.add_row(str(ticker), *[f"{v:.3f}" if pd.notna(v) else "—" for v in row])
    console.print(table)


@app.command("rank")
def cmd_rank(
    tickers: str = typer.Option(_DEFAULT_TICKERS, "--tickers", "-t"),
    signal: str = typer.Option("earnings_yield", "--signal", "-s",
                                help="Signal to rank by"),
    as_of: Optional[str] = typer.Option(None, "--as-of"),
    n_quantiles: int = typer.Option(5, "--quantiles", "-q"),
    db_path: str = typer.Option("data/signals.duckdb", "--db"),
) -> None:
    """Rank the universe by a chosen signal and print quantile breakdown."""
    from signals_engine.signals.fundamentals import all_fundamental_signals
    from signals_engine.signals.momentum import all_momentum_signals
    from signals_engine.rank.crosssection import assign_quantiles, rank_cross_section

    tick_list = _parse_tickers(tickers)
    d = datetime.strptime(as_of, "%Y-%m-%d").date() if as_of else date.today()
    fund = _load_fund(db_path, tick_list)
    prices = _load_prices_wide(db_path, tick_list)

    fund_sigs = all_fundamental_signals(fund, prices, d)
    mom_sigs = all_momentum_signals(prices, d)
    all_sigs = pd.concat([fund_sigs, mom_sigs], axis=1)

    if signal not in all_sigs.columns:
        console.print(f"[red]Signal '{signal}' not found. Available: {list(all_sigs.columns)}[/red]")
        raise typer.Exit(1)

    sig = all_sigs[signal].dropna().sort_values(ascending=False)
    q = assign_quantiles(sig, n=n_quantiles)

    table = Table(title=f"Rankings — {signal} as of {d}", show_lines=True)
    table.add_column("Quantile", style="bold")
    table.add_column("Tickers", style="cyan")
    table.add_column("Mean Value", justify="right")

    for qi in range(n_quantiles, 0, -1):
        members = q[q == qi].index.tolist()
        mean_val = float(sig[members].mean()) if members else float("nan")
        table.add_row(
            f"Q{qi} ({'top' if qi == n_quantiles else 'bottom' if qi == 1 else ''})",
            ", ".join(members),
            f"{mean_val:.3f}",
        )
    console.print(table)


@app.command("backtest")
def cmd_backtest(
    tickers: str = typer.Option(_DEFAULT_TICKERS, "--tickers", "-t"),
    signal: str = typer.Option("earnings_yield", "--signal", "-s"),
    start: str = typer.Option("2019-01-01", "--start"),
    end: Optional[str] = typer.Option(None, "--end"),
    horizon: int = typer.Option(21, "--horizon", "-h",
                                 help="Forward return horizon in trading days"),
    n_quantiles: int = typer.Option(5, "--quantiles", "-q"),
    db_path: str = typer.Option("data/signals.duckdb", "--db"),
) -> None:
    """Run a rolling IC backtest for a single signal and print summary statistics."""
    from signals_engine.data.prices import compute_forward_returns
    from signals_engine.rank.backtest import (
        BacktestResult, build_monthly_signal_panel,
        make_rebalance_dates, run_backtest,
    )
    from signals_engine.signals.fundamentals import (
        earnings_yield, return_on_assets, accruals, leverage, sales_growth, asset_growth,
    )
    from signals_engine.signals.momentum import momentum_12_1

    _FN_MAP = {
        "earnings_yield": earnings_yield,
        "return_on_assets": return_on_assets,
        "accruals": accruals,
        "leverage": leverage,
        "sales_growth": sales_growth,
        "asset_growth": asset_growth,
        "momentum_12_1": momentum_12_1,
    }

    if signal not in _FN_MAP:
        console.print(f"[red]Unknown signal '{signal}'. Options: {list(_FN_MAP.keys())}[/red]")
        raise typer.Exit(1)

    tick_list = _parse_tickers(tickers)
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date() if end else date.today()

    fund = _load_fund(db_path, tick_list)
    prices = _load_prices_wide(db_path, tick_list)
    fwd_ret = compute_forward_returns(prices, horizon=horizon)

    rebal_dates = make_rebalance_dates(start_d, end_d)
    console.print(f"Building signal panel across {len(rebal_dates)} rebalance dates...")
    fn = _FN_MAP[signal]

    if signal == "momentum_12_1":
        sig_panel = build_monthly_signal_panel(fund, prices,
            lambda f, p, d: fn(p, d), rebal_dates)
    else:
        sig_panel = build_monthly_signal_panel(fund, prices,
            lambda f, p, d: fn(f, p, d) if signal == "earnings_yield" else fn(f, d),
            rebal_dates)

    result = run_backtest(sig_panel, fwd_ret, signal, n_quantiles=n_quantiles)

    table = Table(title=f"Backtest Results — {signal}", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Mean IC",          f"{result.mean_ic:.4f}")
    table.add_row("ICIR",             f"{result.icir:.2f}" if pd.notna(result.icir) else "—")
    table.add_row("Hit Rate",         f"{result.hit_rate:.1%}")
    table.add_row("Cum. Spread",      f"{result.cum_spread_return:.1%}")
    table.add_row("Mean Turnover",    f"{result.turnover_mean:.1%}" if pd.notna(result.turnover_mean) else "—")
    table.add_row("Periods",          str(len(result.ic_series.dropna())))
    console.print(table)


if __name__ == "__main__":
    app()
