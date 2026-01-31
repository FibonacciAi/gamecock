#!/usr/bin/env python3
"""
GAMECOCK ANALYZER - Deep Filing Analysis & Visualization
=========================================================
Companion tool for deep analysis of downloaded SEC data.
Provides visualizations, pattern detection, and AI-assisted analysis prompts.
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, List, Dict, Any

# Auto-install dependencies
def install_if_missing(packages):
    missing = []
    for import_name, pip_name in packages:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        import subprocess
        print(f"Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)

install_if_missing([
    ('pandas', 'pandas'),
    ('matplotlib', 'matplotlib'),
    ('numpy', 'numpy'),
    ('rich', 'rich'),
    ('yfinance', 'yfinance'),
])

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

console = Console()

# Configuration
DATA_DIR = Path.home() / "SEC_Data"
FTD_DIR = DATA_DIR / "FTD"
DERIVATIVES_DIR = DATA_DIR / "Derivatives"
ANALYSIS_DIR = DATA_DIR / "Analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATA LOADERS
# ============================================================================

def load_ftd_data(symbol: str, start_date: str = None) -> pd.DataFrame:
    """Load all FTD data for a symbol."""
    import zipfile

    all_data = []

    for ftd_file in FTD_DIR.glob("*"):
        if ftd_file.suffix not in ['.txt', '.zip', '.csv']:
            continue

        try:
            df = None
            if ftd_file.suffix == '.zip':
                with zipfile.ZipFile(ftd_file, 'r') as zf:
                    for name in zf.namelist():
                        if name.endswith(('.txt', '.csv')):
                            with zf.open(name) as f:
                                df = pd.read_csv(f, sep='|', encoding='latin-1',
                                               dtype=str, on_bad_lines='skip')
                            break
            else:
                df = pd.read_csv(ftd_file, sep='|', encoding='latin-1',
                               dtype=str, on_bad_lines='skip')

            if df is not None:
                # Normalize column names (handle 'QUANTITY (FAILS)' vs 'QUANTITY FAIL')
                for col in df.columns:
                    if 'QUANTITY' in col.upper() and 'FAIL' in col.upper():
                        df = df.rename(columns={col: 'QUANTITY FAIL'})
                        break

                if 'SYMBOL' in df.columns:
                    df = df[df['SYMBOL'].str.upper() == symbol.upper()]
                    if not df.empty:
                        all_data.append(df)
        except Exception:
            continue

    if not all_data:
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)

    # Clean and convert
    combined['QUANTITY FAIL'] = pd.to_numeric(combined['QUANTITY FAIL'], errors='coerce')
    combined['PRICE'] = pd.to_numeric(combined['PRICE'], errors='coerce')
    combined['SETTLEMENT DATE'] = pd.to_datetime(combined['SETTLEMENT DATE'], format='%Y%m%d', errors='coerce')

    if start_date:
        combined = combined[combined['SETTLEMENT DATE'] >= start_date]

    combined = combined.sort_values('SETTLEMENT DATE')
    combined = combined.drop_duplicates(subset=['SETTLEMENT DATE', 'CUSIP'])

    return combined

def load_price_data(symbol: str, start_date: str, end_date: str = None) -> pd.DataFrame:
    """Load historical price data from Yahoo Finance."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date or datetime.now().strftime('%Y-%m-%d'))
        hist.index = hist.index.tz_localize(None)  # Remove timezone
        return hist
    except Exception as e:
        console.print(f"[yellow]Could not load price data: {e}[/yellow]")
        return pd.DataFrame()

def load_swap_data(symbol: str) -> pd.DataFrame:
    """Load swap/derivative data for a symbol."""
    all_data = []

    for swap_dir in [DERIVATIVES_DIR / "equity", DERIVATIVES_DIR / "credit"]:
        if not swap_dir.exists():
            continue

        for csv_file in swap_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, dtype=str, on_bad_lines='skip')

                # Search in various columns
                for col in ['Underlier ID', 'Reference Entity', 'underlying_asset', 'Underlying Asset']:
                    if col in df.columns:
                        mask = df[col].str.contains(symbol, case=False, na=False)
                        filtered = df[mask]
                        if not filtered.empty:
                            all_data.append(filtered)
                        break
            except Exception:
                continue

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def calculate_ftd_metrics(df: pd.DataFrame) -> Dict:
    """Calculate key FTD metrics."""
    if df.empty:
        return {}

    return {
        'total_fails': int(df['QUANTITY FAIL'].sum()),
        'avg_daily_fails': float(df['QUANTITY FAIL'].mean()),
        'max_fails': int(df['QUANTITY FAIL'].max()),
        'max_fails_date': df.loc[df['QUANTITY FAIL'].idxmax(), 'SETTLEMENT DATE'],
        'std_dev': float(df['QUANTITY FAIL'].std()),
        'days_with_fails': len(df),
        'unique_dates': df['SETTLEMENT DATE'].nunique(),
    }

def detect_ftd_anomalies(df: pd.DataFrame, threshold_std: float = 2.5) -> pd.DataFrame:
    """Detect anomalous FTD spikes."""
    if df.empty:
        return pd.DataFrame()

    mean = df['QUANTITY FAIL'].mean()
    std = df['QUANTITY FAIL'].std()

    if std == 0:
        return pd.DataFrame()

    threshold = mean + (threshold_std * std)
    anomalies = df[df['QUANTITY FAIL'] > threshold].copy()
    anomalies['z_score'] = (anomalies['QUANTITY FAIL'] - mean) / std

    return anomalies.sort_values('z_score', ascending=False)

def detect_price_ftd_correlation(ftd_df: pd.DataFrame, price_df: pd.DataFrame) -> Dict:
    """Analyze correlation between FTD spikes and price movements."""
    if ftd_df.empty or price_df.empty:
        return {'correlation': None, 'findings': []}

    # Merge on date
    ftd_daily = ftd_df.groupby(ftd_df['SETTLEMENT DATE'].dt.date)['QUANTITY FAIL'].sum().reset_index()
    ftd_daily.columns = ['Date', 'FTD']
    ftd_daily['Date'] = pd.to_datetime(ftd_daily['Date'])

    price_df = price_df.reset_index()
    price_df.columns = ['Date'] + list(price_df.columns[1:])
    price_df['Date'] = pd.to_datetime(price_df['Date'])

    merged = pd.merge(ftd_daily, price_df[['Date', 'Close', 'Volume']], on='Date', how='inner')

    if len(merged) < 10:
        return {'correlation': None, 'findings': ['Insufficient overlapping data']}

    # Calculate correlations
    ftd_price_corr = merged['FTD'].corr(merged['Close'])
    ftd_volume_corr = merged['FTD'].corr(merged['Volume'])

    # Calculate price change following high FTD days
    ftd_threshold = merged['FTD'].quantile(0.9)
    high_ftd_days = merged[merged['FTD'] > ftd_threshold].index

    price_changes_after_high_ftd = []
    for idx in high_ftd_days:
        if idx + 5 < len(merged):
            change = (merged.loc[idx + 5, 'Close'] - merged.loc[idx, 'Close']) / merged.loc[idx, 'Close']
            price_changes_after_high_ftd.append(change)

    findings = []
    if abs(ftd_price_corr) > 0.3:
        direction = "positive" if ftd_price_corr > 0 else "negative"
        findings.append(f"Moderate {direction} correlation ({ftd_price_corr:.2f}) between FTD and price")

    if price_changes_after_high_ftd:
        avg_change = np.mean(price_changes_after_high_ftd)
        if abs(avg_change) > 0.05:
            direction = "increase" if avg_change > 0 else "decrease"
            findings.append(f"Average {direction} of {abs(avg_change)*100:.1f}% within 5 days of high FTD")

    return {
        'ftd_price_correlation': ftd_price_corr,
        'ftd_volume_correlation': ftd_volume_corr,
        'avg_price_change_after_high_ftd': np.mean(price_changes_after_high_ftd) if price_changes_after_high_ftd else None,
        'findings': findings,
    }

def calculate_settlement_pressure(ftd_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate T+35 and REG SHO settlement pressure dates."""
    if ftd_df.empty:
        return pd.DataFrame()

    # T+35 rule: FTDs must be closed within 35 calendar days
    # REG SHO threshold: 10,000 shares or 0.5% of shares outstanding

    pressure_dates = []
    for _, row in ftd_df.iterrows():
        fail_date = row['SETTLEMENT DATE']
        fail_qty = row['QUANTITY FAIL']

        if pd.isna(fail_date) or pd.isna(fail_qty):
            continue

        # T+35 settlement deadline
        t35_date = fail_date + timedelta(days=35)

        pressure_dates.append({
            'fail_date': fail_date,
            'settlement_deadline': t35_date,
            'quantity': fail_qty,
            'pressure_type': 'T+35',
        })

    if pressure_dates:
        df = pd.DataFrame(pressure_dates)
        # Aggregate by settlement deadline
        pressure_by_date = df.groupby('settlement_deadline')['quantity'].sum().reset_index()
        pressure_by_date.columns = ['date', 'settlement_pressure']
        return pressure_by_date

    return pd.DataFrame()

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ftd_analysis(symbol: str, ftd_df: pd.DataFrame, price_df: pd.DataFrame,
                      save_path: Path = None):
    """Create comprehensive FTD analysis chart."""
    if ftd_df.empty:
        console.print("[yellow]No FTD data to plot[/yellow]")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{symbol} - FTD Analysis', fontsize=14, fontweight='bold')

    # Aggregate FTD by date
    ftd_daily = ftd_df.groupby(ftd_df['SETTLEMENT DATE'].dt.date)['QUANTITY FAIL'].sum()
    ftd_daily.index = pd.to_datetime(ftd_daily.index)

    # Plot 1: FTD volume
    ax1 = axes[0]
    ax1.bar(ftd_daily.index, ftd_daily.values, color='crimson', alpha=0.7, label='FTD Volume')

    # Add anomaly markers
    anomalies = detect_ftd_anomalies(ftd_df)
    if not anomalies.empty:
        anomaly_dates = anomalies['SETTLEMENT DATE']
        anomaly_vals = anomalies.groupby(anomalies['SETTLEMENT DATE'].dt.date)['QUANTITY FAIL'].sum()
        ax1.scatter(anomaly_dates, anomaly_vals.values, color='yellow', s=100,
                   marker='*', zorder=5, label='Anomalies')

    ax1.set_ylabel('FTD Shares')
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Price (if available)
    ax2 = axes[1]
    if not price_df.empty:
        ax2.plot(price_df.index, price_df['Close'], color='blue', linewidth=1.5, label='Close Price')
        ax2.fill_between(price_df.index, price_df['Low'], price_df['High'], alpha=0.2, color='blue')
        ax2.set_ylabel('Price ($)')
        ax2.legend(loc='upper right')
    else:
        ax2.text(0.5, 0.5, 'Price data unavailable', transform=ax2.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Settlement pressure (T+35)
    ax3 = axes[2]
    pressure = calculate_settlement_pressure(ftd_df)
    if not pressure.empty:
        ax3.bar(pressure['date'], pressure['settlement_pressure'],
               color='purple', alpha=0.6, label='T+35 Settlement Pressure')
        ax3.set_ylabel('Settlement Pressure')
        ax3.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
        ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('Date')

    # Format x-axis
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]Chart saved to: {save_path}[/green]")

    plt.show()

def plot_swap_timeline(symbol: str, swap_df: pd.DataFrame, save_path: Path = None):
    """Plot swap transaction timeline."""
    if swap_df.empty:
        console.print("[yellow]No swap data to plot[/yellow]")
        return

    # Try to parse execution timestamp
    time_col = None
    for col in ['Execution Timestamp', 'execution_timestamp', 'Event timestamp']:
        if col in swap_df.columns:
            time_col = col
            break

    if not time_col:
        console.print("[yellow]No timestamp column found in swap data[/yellow]")
        return

    swap_df['exec_time'] = pd.to_datetime(swap_df[time_col], errors='coerce')
    swap_df = swap_df.dropna(subset=['exec_time'])

    if swap_df.empty:
        console.print("[yellow]No valid timestamps in swap data[/yellow]")
        return

    # Aggregate by date
    daily_counts = swap_df.groupby(swap_df['exec_time'].dt.date).size()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(pd.to_datetime(daily_counts.index), daily_counts.values, color='teal', alpha=0.7)
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Transactions')
    ax.set_title(f'{symbol} - Swap Transaction Timeline')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        console.print(f"[green]Chart saved to: {save_path}[/green]")

    plt.show()

# ============================================================================
# AI ANALYSIS PROMPTS
# ============================================================================

def generate_analysis_prompt(symbol: str, metrics: Dict, anomalies: pd.DataFrame,
                             correlation: Dict) -> str:
    """Generate a prompt for AI-assisted analysis."""
    prompt = f"""
ANALYZE THE FOLLOWING SEC DATA FOR {symbol}:

## FTD METRICS
- Total Failed Shares: {metrics.get('total_fails', 'N/A'):,}
- Average Daily Fails: {metrics.get('avg_daily_fails', 'N/A'):,.0f}
- Maximum Single-Day Fails: {metrics.get('max_fails', 'N/A'):,} on {metrics.get('max_fails_date', 'N/A')}
- Standard Deviation: {metrics.get('std_dev', 'N/A'):,.0f}
- Days with FTD Records: {metrics.get('days_with_fails', 'N/A')}

## ANOMALIES DETECTED
"""
    if not anomalies.empty:
        for _, row in anomalies.head(10).iterrows():
            prompt += f"- {row['SETTLEMENT DATE'].strftime('%Y-%m-%d')}: {int(row['QUANTITY FAIL']):,} shares (Z-score: {row['z_score']:.2f})\n"
    else:
        prompt += "No significant anomalies detected.\n"

    prompt += f"""
## PRICE CORRELATION
- FTD-Price Correlation: {correlation.get('ftd_price_correlation', 'N/A')}
- FTD-Volume Correlation: {correlation.get('ftd_volume_correlation', 'N/A')}
- Avg Price Change After High FTD: {correlation.get('avg_price_change_after_high_ftd', 'N/A')}

## KEY FINDINGS
"""
    for finding in correlation.get('findings', []):
        prompt += f"- {finding}\n"

    prompt += """
## ANALYSIS QUESTIONS
1. What patterns suggest potential market manipulation or naked short selling?
2. Are the FTD spikes correlated with any known market events?
3. Does the T+35 settlement pressure align with any price movements?
4. What regulatory filings should be examined for these anomalous periods?
5. Are there any coordinated activities suggested by the data?

Please provide a detailed analysis focusing on:
- Potential manipulation indicators
- Regulatory compliance concerns
- Timeline of suspicious activities
- Recommended further investigation
"""
    return prompt

# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def analyze_symbol_interactive():
    """Interactive symbol analysis."""
    symbol = Prompt.ask("Enter stock symbol").upper()
    start_date = Prompt.ask("Start date (YYYY-MM-DD)", default="2020-01-01")

    console.print(f"\n[cyan]Loading data for {symbol}...[/cyan]")

    # Load FTD data
    ftd_df = load_ftd_data(symbol, start_date)

    if ftd_df.empty:
        console.print("[yellow]No FTD data found. Have you downloaded FTD files?[/yellow]")
        return

    console.print(f"[green]Loaded {len(ftd_df)} FTD records[/green]")

    # Load price data
    price_df = load_price_data(symbol, start_date)
    if not price_df.empty:
        console.print(f"[green]Loaded {len(price_df)} price records[/green]")

    # Calculate metrics
    metrics = calculate_ftd_metrics(ftd_df)

    # Detect anomalies
    anomalies = detect_ftd_anomalies(ftd_df)

    # Calculate correlation
    correlation = detect_price_ftd_correlation(ftd_df, price_df)

    # Display results
    console.print(Panel(f"[bold]{symbol} FTD Analysis[/bold]", border_style="blue"))

    metrics_table = Table(show_header=True, header_style="bold cyan")
    metrics_table.add_column("Metric")
    metrics_table.add_column("Value", justify="right")

    metrics_table.add_row("Total Failed Shares", f"{metrics.get('total_fails', 0):,}")
    metrics_table.add_row("Average Daily Fails", f"{metrics.get('avg_daily_fails', 0):,.0f}")
    metrics_table.add_row("Maximum Single-Day", f"{metrics.get('max_fails', 0):,}")
    metrics_table.add_row("Max Fails Date", str(metrics.get('max_fails_date', 'N/A'))[:10])
    metrics_table.add_row("Standard Deviation", f"{metrics.get('std_dev', 0):,.0f}")
    metrics_table.add_row("Days with Records", str(metrics.get('days_with_fails', 0)))

    console.print(metrics_table)

    # Anomalies
    if not anomalies.empty:
        console.print(f"\n[red]ANOMALIES DETECTED: {len(anomalies)}[/red]")

        anom_table = Table(show_header=True, header_style="bold red")
        anom_table.add_column("Date")
        anom_table.add_column("Quantity", justify="right")
        anom_table.add_column("Z-Score", justify="right")

        for _, row in anomalies.head(10).iterrows():
            anom_table.add_row(
                str(row['SETTLEMENT DATE'])[:10],
                f"{int(row['QUANTITY FAIL']):,}",
                f"{row['z_score']:.2f}"
            )

        console.print(anom_table)

    # Correlation findings
    if correlation.get('findings'):
        console.print("\n[bold magenta]CORRELATION FINDINGS:[/bold magenta]")
        for finding in correlation['findings']:
            console.print(f"  • {finding}")

    # Visualization
    if Confirm.ask("\nGenerate visualization?"):
        save_path = ANALYSIS_DIR / f"{symbol}_ftd_analysis_{datetime.now().strftime('%Y%m%d')}.png"
        plot_ftd_analysis(symbol, ftd_df, price_df, save_path)

    # AI prompt
    if Confirm.ask("Generate AI analysis prompt?"):
        prompt = generate_analysis_prompt(symbol, metrics, anomalies, correlation)
        prompt_path = ANALYSIS_DIR / f"{symbol}_analysis_prompt_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(prompt_path, 'w') as f:
            f.write(prompt)
        console.print(f"\n[green]Prompt saved to: {prompt_path}[/green]")
        console.print("\n[dim]Copy this prompt to Claude or another AI for detailed analysis:[/dim]")
        console.print(Panel(prompt[:1000] + "...", title="Analysis Prompt Preview"))

def compare_symbols_interactive():
    """Compare FTD patterns across multiple symbols."""
    symbols_input = Prompt.ask("Enter symbols (comma-separated)")
    symbols = [s.strip().upper() for s in symbols_input.split(',')]
    start_date = Prompt.ask("Start date (YYYY-MM-DD)", default="2020-01-01")

    all_metrics = {}

    for symbol in symbols:
        console.print(f"[cyan]Analyzing {symbol}...[/cyan]")
        ftd_df = load_ftd_data(symbol, start_date)
        if not ftd_df.empty:
            all_metrics[symbol] = calculate_ftd_metrics(ftd_df)
        else:
            console.print(f"[yellow]No data for {symbol}[/yellow]")

    if not all_metrics:
        console.print("[red]No data found for any symbols[/red]")
        return

    # Comparison table
    comp_table = Table(show_header=True, header_style="bold cyan")
    comp_table.add_column("Symbol")
    comp_table.add_column("Total Fails", justify="right")
    comp_table.add_column("Avg Daily", justify="right")
    comp_table.add_column("Max Single Day", justify="right")
    comp_table.add_column("Max Date")

    for symbol, metrics in all_metrics.items():
        comp_table.add_row(
            symbol,
            f"{metrics.get('total_fails', 0):,}",
            f"{metrics.get('avg_daily_fails', 0):,.0f}",
            f"{metrics.get('max_fails', 0):,}",
            str(metrics.get('max_fails_date', 'N/A'))[:10]
        )

    console.print(comp_table)

def main():
    """Main entry point."""
    console.print(Panel(
        "[bold blue]GAMECOCK ANALYZER[/bold blue]\n"
        "[dim]Deep Filing Analysis & Visualization[/dim]",
        border_style="blue"
    ))

    while True:
        console.print("\n[bold]Options:[/bold]")
        console.print("  1. Analyze single symbol")
        console.print("  2. Compare multiple symbols")
        console.print("  3. Load and analyze swap data")
        console.print("  q. Quit")

        choice = Prompt.ask("\nSelect option", default="q")

        if choice == '1':
            analyze_symbol_interactive()
        elif choice == '2':
            compare_symbols_interactive()
        elif choice == '3':
            symbol = Prompt.ask("Enter symbol")
            swap_df = load_swap_data(symbol)
            if swap_df.empty:
                console.print("[yellow]No swap data found. Download from DTCC first.[/yellow]")
            else:
                console.print(f"[green]Found {len(swap_df)} swap records[/green]")
                console.print(swap_df.head(20).to_string())
                if Confirm.ask("Plot timeline?"):
                    plot_swap_timeline(symbol, swap_df)
        elif choice.lower() == 'q':
            break

if __name__ == "__main__":
    main()
