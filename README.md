# GAMECOCK - SEC Filing Truth Finder

**Live Dashboard:** [fibonacciai.github.io/gamecock](https://fibonacciai.github.io/gamecock/)

An enhanced SEC data aggregation and analysis tool focused on **uncovering patterns, anomalies, and truth** within financial filings. Built for financial forensics and deep investigation of market mechanics.

## What This Project Does

GAMECOCK downloads, processes, and cross-references SEC data to identify potential market manipulation, settlement failures, and suspicious trading patterns. It includes a complete deep-dive investigation into GameStop (GME) as a case study.

### Core Tools

| File | Purpose |
|------|---------|
| `gamecock.py` | Main CLI tool - SEC scraping, FTD analysis, insider patterns, Truth Finder |
| `analyzer.py` | Deep analysis - visualizations, price correlation, AI prompts |
| `index.html` | GME Investigation Dashboard - interactive report with all findings |
| `gamecock.command` | macOS double-click launcher |

## Features

### Data Collection
- **SEC EDGAR Filings** - Search and download 10-K, 10-Q, 8-K, 13F, Form 4, and 18+ form types
- **Fails-to-Deliver (FTD)** - Download and analyze FTD data from 2004-present (137+ files)
- **DTCC Derivatives** - Track equity and credit swap transactions
- **Insider Trading** - Form 3/4/5 pattern detection with transaction code parsing
- **Full Index Downloads** - Quarterly EDGAR master indexes

### Analysis Capabilities
- **Anomaly Detection** - Z-score analysis to find FTD spikes (>2.5 std dev threshold)
- **Cross-Reference Analysis** - Correlate FTD spikes with insider selling dates
- **Price Correlation** - Analyze FTD-to-price relationships using Yahoo Finance data
- **T+35 Settlement Pressure** - Calculate REG SHO settlement deadline pressure
- **Coordinated Activity Detection** - Find patterns in insider transactions

### Truth Finder Mode
Full cross-source analysis that generates a **Suspicion Score (0-100)** based on:
- FTD anomalies and patterns
- Derivative activity spikes
- Insider trading coordination
- Cross-reference correlations between data sources

## GME Deep Investigation

The included dashboard (`index.html`) contains a comprehensive GameStop analysis:

### Historical FTD Analysis (2020-2025)
| Metric | Value |
|--------|-------|
| Total FTD Shares | **81.9 million** |
| January 2021 FTD | **15.7 million** (single month) |
| Pre-Squeeze Q4 2020 | **22.6 million** |
| Anomaly Days Detected | **18** (>2.5 std dev) |
| Max Single Day | **3.21M shares** (Oct 13, 2020) |
| Price Change After High FTD | **+30.9%** avg within 5 days |

### Current State (January 2026)
| Metric | Value |
|--------|-------|
| Cash Position | **$7.84 billion** |
| Total Assets | **$10.55 billion** |
| Market Cap | **$10.2 billion** |
| Current Price | ~$22.81 |
| Q3 2025 Net Income | **$77.1M** (profitable) |
| Short Interest | **16.1%** of float |
| FTD Status | **ZERO** since Sept 2025 |

### October 2025 Warrant Dividend
| Detail | Value |
|--------|-------|
| Warrants Issued | **59.15 million** |
| Exercise Price | **$32.00** |
| Ticker | **GME WS** (NYSE) |
| Expiration | **Oct 30, 2026** |
| Potential Proceeds | **$1.9 billion** |
| Distribution | 1 warrant per 10 shares |

### Key Findings

1. **FTD Accumulation Preceded Squeeze** - 22.6M shares failed in Q4 2020, creating settlement pressure that exploded in January 2021

2. **T+35 Correlation** - December 2020 FTDs required settlement by late January 2021, precisely when the squeeze occurred

3. **Statistical Anomalies Cluster** - 18 days exceeded 2.5 standard deviations, with highest at 9.4 std dev (Oct 13, 2020)

4. **Settlement Crisis Resolved** - Zero GME FTD records in SEC data since September 2025

5. **New Warrant Variable** - 59M warrants at $32 strike create new battlefield; warrant FTD data available ~Feb 2026

## Installation

```bash
git clone https://github.com/FibonacciAi/gamecock.git
cd gamecock
python3 gamecock.py
```

Dependencies auto-install on first run:
- pandas, requests, beautifulsoup4, lxml, chardet, rich, matplotlib, yfinance

## Usage

### Interactive CLI
```bash
python3 gamecock.py
```

**Menu Options:**
| # | Action |
|---|--------|
| 1 | Search SEC EDGAR filings |
| 2 | Download FTD data |
| 3 | Search FTD records by symbol |
| 4 | Analyze insider trading patterns |
| 5 | Search derivatives/swaps |
| 6 | **TRUTH FINDER** - Full cross-reference analysis |
| 7 | Download EDGAR quarterly index |
| 8 | Form type reference guide |

### Deep Analysis
```bash
python3 analyzer.py
```
- FTD trend visualization with price overlay
- T+35 settlement pressure charts
- Price-FTD correlation analysis
- AI analysis prompt generation
- Multi-symbol comparison

### View Dashboard
Open `index.html` in browser or visit [fibonacciai.github.io/gamecock](https://fibonacciai.github.io/gamecock/)

## Data Storage

All data stored in `~/SEC_Data/`:
```
~/SEC_Data/
├── EDGAR/          # SEC filings
├── FTD/            # Fails-to-deliver data (137+ files)
├── Derivatives/    # Swap/derivative data
├── Insider/        # Form 4 filings
├── Exchange/       # Exchange volume data
├── Analysis/       # Generated reports & charts
└── gamecock.db     # SQLite tracking database
```

## Technical Details

### FTD Data Processing
- Handles multiple SEC file formats (2004-present schema changes)
- Normalizes column names (`QUANTITY (FAILS)` vs `QUANTITY FAIL`)
- Deduplicates by settlement date + CUSIP
- Supports ZIP and TXT formats

### SEC API Compliance
- 10 requests/second rate limiting enforced
- Proper User-Agent headers
- Handles 403/404 gracefully

### Analysis Methods
- Z-score anomaly detection (configurable threshold)
- Pearson correlation coefficients
- Rolling statistics for trend detection
- T+35 calendar day calculations

## Key Concepts

### Fails-to-Deliver (FTD)
When a seller fails to deliver shares by settlement date. High FTDs indicate:
- Naked short selling
- Operational settlement failures
- Potential market manipulation

### T+35 Rule
Market makers must close FTD positions within 35 calendar days under REG SHO. Settlement pressure often correlates with price movements.

### Form 4 Insider Trading
Required within 2 business days when insiders trade. Transaction codes:
- `P` = Purchase, `S` = Sale, `M` = Option exercise
- `A` = Grant/Award, `D` = Disposition, `G` = Gift

## Red Flags Detected

| Flag Type | Severity | Trigger |
|-----------|----------|---------|
| `anomalous_fails` | HIGH | FTD >3 std deviations |
| `coordinated_swaps` | HIGH | Clustered derivative activity |
| `coordinated_selling` | MEDIUM | Multiple insiders same day |
| `ftd_insider_correlation` | CRITICAL | FTD spike + insider selling |

## SEC Filing Evidence Links

- [GameStop All Filings](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001326380)
- [Form 4 Insider Trades](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001326380&type=4&owner=only)
- [8-K Material Events](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0001326380&type=8-K)
- [SEC FTD Data Files](https://www.sec.gov/data/foiadocsfailsdatahtm)
- [Warrant Press Release](https://investor.gamestop.com/news-releases/news-details/2025/GameStop-Announces-the-Distribution-of-Warrants-to-Shareholders/)

## Legal Disclaimer

This tool is for **educational and research purposes only**.

- Comply with SEC data usage policies
- Not financial or investment advice
- DTCC data requires separate registration
- Verify findings independently

## Credits

Created by [@D74169](https://x.com/D74169)

Inspired by [SECthingv2](https://github.com/artpersonnft/SECthingv2) by artpersonnft.

Built with Claude for deep financial forensics and truth-finding.

---

*"The market can remain irrational longer than you can remain solvent." — In GameStop's case, the irrationality became the solvency.*
