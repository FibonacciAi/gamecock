# GAMECOCK - SEC Filing Truth Finder

An enhanced SEC data aggregation and analysis tool focused on **uncovering patterns, anomalies, and truth** within financial filings.

## Features

### Data Collection
- **SEC EDGAR Filings** - Search and download 10-K, 10-Q, 8-K, 13F, Form 4, and more
- **Fails-to-Deliver (FTD)** - Download and analyze FTD data from 2004-present
- **DTCC Derivatives** - Track equity and credit swap transactions
- **Insider Trading** - Form 3/4/5 pattern detection
- **Full Index Downloads** - Quarterly EDGAR master indexes

### Analysis Capabilities
- **Anomaly Detection** - Statistical analysis to find FTD spikes (>2.5 std dev)
- **Cross-Reference Analysis** - Correlate FTD spikes with insider selling
- **Price Correlation** - Analyze FTD-to-price relationships
- **T+35 Settlement Pressure** - Calculate settlement deadline pressure
- **Coordinated Activity Detection** - Find patterns in insider transactions

### Truth Finder Mode
Full cross-source analysis that generates a **Suspicion Score (0-100)** based on:
- FTD anomalies and patterns
- Derivative activity spikes
- Insider trading coordination
- Cross-reference correlations

## Installation

```bash
cd gamecock
python3 gamecock.py
```

Dependencies are automatically installed on first run:
- pandas
- requests
- beautifulsoup4
- lxml
- chardet
- rich
- matplotlib

## Usage

### Quick Start
```bash
# macOS: double-click
./gamecock.command

# Or run directly
python3 gamecock.py
```

### Main Menu Options

| Option | Description |
|--------|-------------|
| 1 | Search SEC EDGAR filings |
| 2 | Download FTD data |
| 3 | Search FTD records |
| 4 | Analyze insider trading |
| 5 | Search derivatives/swaps |
| 6 | **TRUTH FINDER** - Full analysis |
| 7 | Download EDGAR index |
| 8 | Form type reference |

### Example: Analyze a Stock

1. Download FTD data (option 2)
2. Run Truth Finder (option 6)
3. Enter stock symbol (e.g., GME)
4. Enter CIK for insider analysis
5. Review Suspicion Score and red flags

## Deep Analysis (analyzer.py)

For advanced visualization and AI-assisted analysis:

```bash
python3 analyzer.py
```

Features:
- FTD trend visualization with price overlay
- T+35 settlement pressure charts
- Price-FTD correlation analysis
- AI analysis prompt generation (for Claude/GPT)
- Multi-symbol comparison

## Data Storage

All data is stored in `~/SEC_Data/`:
```
~/SEC_Data/
├── EDGAR/          # SEC filings
├── FTD/            # Fails-to-deliver data
├── Derivatives/    # Swap/derivative data
├── Insider/        # Form 4 filings
├── Exchange/       # Exchange volume data
├── Analysis/       # Generated reports
└── gamecock.db     # SQLite database
```

## Key Concepts

### Fails-to-Deliver (FTD)
When a seller fails to deliver shares to the buyer by settlement date. High FTDs can indicate:
- Naked short selling
- Operational failures
- Market manipulation

### T+35 Rule
Market makers must close FTD positions within 35 calendar days. Settlement pressure often correlates with price movements.

### Form 4 Insider Trading
Required filing within 2 business days when insiders buy/sell company stock. Patterns to watch:
- Coordinated selling before bad news
- Cluster transactions
- Options exercises followed by sales

## Red Flags Detected

| Flag Type | Severity | Description |
|-----------|----------|-------------|
| `anomalous_fails` | HIGH | FTD spikes >3 std deviations |
| `coordinated_swaps` | HIGH | Clustered derivative activity |
| `coordinated_selling` | MEDIUM | Multiple insiders selling same day |
| `ftd_insider_correlation` | CRITICAL | FTD spike + insider selling |

## Legal Disclaimer

This tool is for **educational and research purposes only**.

- Always comply with SEC data usage policies
- 10 requests/second rate limit is enforced
- DTCC data requires separate registration
- Not financial advice

## Credits

Created by [@D74169](https://x.com/D74169)

Inspired by [SECthingv2](https://github.com/artpersonnft/SECthingv2) by artpersonnft.

Enhanced for deep financial forensics and truth-finding with Claude.
