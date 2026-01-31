#!/usr/bin/env python3
"""
GAMECOCK - SEC Filing Truth Finder
==================================
An enhanced SEC data aggregation and analysis tool focused on
uncovering patterns, anomalies, and truth within financial filings.

Features:
- SEC EDGAR filing downloads (10-K, 10-Q, 8-K, 13F, Form 4, etc.)
- Fails-to-Deliver (FTD) data analysis
- DTCC derivatives (credit/equity swaps) tracking
- Insider trading pattern detection
- Cross-reference analysis for connected entities
- Anomaly detection and red flag identification

Original inspiration: https://github.com/artpersonnft/SECthingv2
Enhanced for deep financial forensics.
"""

import os
import sys
import csv
import json
import hashlib
import re
import time
import zipfile
import io
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple
import urllib.request
import urllib.parse

# Auto-install dependencies
REQUIRED_PACKAGES = [
    ('pandas', 'pandas'),
    ('requests', 'requests'),
    ('bs4', 'beautifulsoup4'),
    ('lxml', 'lxml'),
    ('chardet', 'chardet'),
    ('rich', 'rich'),
    ('matplotlib', 'matplotlib'),
]

def install_dependencies():
    """Install required packages if missing."""
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)
        print("Dependencies installed. Restarting...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

install_dependencies()

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown

# Initialize Rich console
console = Console()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for Gamecock."""

    # Base directories
    BASE_DIR = Path.home() / "SEC_Data"
    EDGAR_DIR = BASE_DIR / "EDGAR"
    FTD_DIR = BASE_DIR / "FTD"
    DERIVATIVES_DIR = BASE_DIR / "Derivatives"
    INSIDER_DIR = BASE_DIR / "Insider"
    EXCHANGE_DIR = BASE_DIR / "Exchange"
    ANALYSIS_DIR = BASE_DIR / "Analysis"
    DB_PATH = BASE_DIR / "gamecock.db"

    # Request settings
    USER_AGENT = "Gamecock/1.0 (Financial Research; Contact: researcher@example.com)"
    REQUEST_DELAY = 0.1  # SEC rate limit compliance
    MAX_WORKERS = 8
    TIMEOUT = 30

    # SEC URLs
    SEC_BASE = "https://www.sec.gov"
    EDGAR_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"
    EDGAR_FULL_INDEX = "https://www.sec.gov/Archives/edgar/full-index"
    FTD_BASE = "https://www.sec.gov/data/foiadocsfailsdatahtm"
    DTCC_EQUITY = "https://pddata.dtcc.com/ppd/secdashboard"
    DTCC_CREDIT = "https://pddata.dtcc.com/ppd/crdashboard"

    @classmethod
    def ensure_dirs(cls):
        """Create all necessary directories."""
        for d in [cls.EDGAR_DIR, cls.FTD_DIR, cls.DERIVATIVES_DIR,
                  cls.INSIDER_DIR, cls.EXCHANGE_DIR, cls.ANALYSIS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

Config.ensure_dirs()

# ============================================================================
# DATABASE LAYER
# ============================================================================

class Database:
    """SQLite database for tracking and analysis."""

    def __init__(self, db_path: Path = Config.DB_PATH):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS filings (
                id INTEGER PRIMARY KEY,
                cik TEXT,
                company TEXT,
                form_type TEXT,
                date_filed TEXT,
                accession TEXT UNIQUE,
                url TEXT,
                downloaded INTEGER DEFAULT 0,
                analyzed INTEGER DEFAULT 0,
                flags TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ftd_records (
                id INTEGER PRIMARY KEY,
                settlement_date TEXT,
                cusip TEXT,
                symbol TEXT,
                quantity INTEGER,
                description TEXT,
                price REAL,
                source_file TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS derivatives (
                id INTEGER PRIMARY KEY,
                dissemination_id TEXT UNIQUE,
                action TEXT,
                execution_timestamp TEXT,
                underlying_asset TEXT,
                notional_amount REAL,
                notional_currency TEXT,
                price REAL,
                counterparty TEXT,
                source_type TEXT,
                source_file TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS insiders (
                id INTEGER PRIMARY KEY,
                cik TEXT,
                issuer_cik TEXT,
                issuer_name TEXT,
                owner_name TEXT,
                form_type TEXT,
                transaction_date TEXT,
                transaction_code TEXT,
                shares REAL,
                price REAL,
                ownership_type TEXT,
                accession TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS red_flags (
                id INTEGER PRIMARY KEY,
                entity TEXT,
                flag_type TEXT,
                severity TEXT,
                description TEXT,
                evidence TEXT,
                related_filings TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_filings_cik ON filings(cik);
            CREATE INDEX IF NOT EXISTS idx_filings_form ON filings(form_type);
            CREATE INDEX IF NOT EXISTS idx_ftd_symbol ON ftd_records(symbol);
            CREATE INDEX IF NOT EXISTS idx_ftd_cusip ON ftd_records(cusip);
            CREATE INDEX IF NOT EXISTS idx_derivatives_underlying ON derivatives(underlying_asset);
            CREATE INDEX IF NOT EXISTS idx_insiders_issuer ON insiders(issuer_cik);
        """)
        self.conn.commit()

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def executemany(self, sql: str, params: list) -> sqlite3.Cursor:
        return self.conn.executemany(sql, params)

    def commit(self):
        self.conn.commit()

# Global database instance
db = Database()

# ============================================================================
# HTTP CLIENT
# ============================================================================

class SECClient:
    """HTTP client with SEC-compliant rate limiting."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': Config.USER_AGENT,
            'Accept-Encoding': 'gzip, deflate',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        self.last_request = 0

    def _rate_limit(self):
        """Enforce SEC rate limiting (10 requests/second max)."""
        elapsed = time.time() - self.last_request
        if elapsed < Config.REQUEST_DELAY:
            time.sleep(Config.REQUEST_DELAY - elapsed)
        self.last_request = time.time()

    def get(self, url: str, **kwargs) -> requests.Response:
        """GET request with rate limiting."""
        self._rate_limit()
        kwargs.setdefault('timeout', Config.TIMEOUT)
        return self.session.get(url, **kwargs)

    def download_file(self, url: str, dest: Path) -> bool:
        """Download file to destination."""
        try:
            self._rate_limit()
            response = self.session.get(url, timeout=Config.TIMEOUT, stream=True)
            response.raise_for_status()

            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            console.print(f"[red]Download failed: {url} - {e}[/red]")
            return False

client = SECClient()

# ============================================================================
# SEC EDGAR OPERATIONS
# ============================================================================

class EDGARScraper:
    """SEC EDGAR filing scraper."""

    FORM_TYPES = {
        '10-K': 'Annual report',
        '10-Q': 'Quarterly report',
        '8-K': 'Current report (material events)',
        '13F-HR': 'Institutional holdings',
        '13F-NT': 'Institutional holdings notice',
        '13D': 'Beneficial ownership (>5%)',
        '13G': 'Beneficial ownership (passive)',
        'SC 13D': 'Schedule 13D',
        'SC 13G': 'Schedule 13G',
        '4': 'Insider transaction',
        '3': 'Initial insider ownership',
        '5': 'Annual insider ownership',
        'DEF 14A': 'Proxy statement',
        'S-1': 'Registration statement',
        '424B': 'Prospectus',
        '6-K': 'Foreign issuer report',
        '20-F': 'Foreign annual report',
        'N-PORT': 'Monthly portfolio holdings',
    }

    def search_company(self, query: str, form_type: str = None) -> List[Dict]:
        """Search for company filings by name or CIK."""
        params = {
            'action': 'getcompany',
            'output': 'atom',
            'count': 100,
        }

        if query.isdigit():
            params['CIK'] = query
        else:
            params['company'] = query

        if form_type:
            params['type'] = form_type

        try:
            response = client.get(Config.EDGAR_BASE, params=params)
            soup = BeautifulSoup(response.content, 'lxml-xml')

            results = []
            for entry in soup.find_all('entry'):
                filing = {
                    'title': entry.find('title').text if entry.find('title') else '',
                    'link': entry.find('link')['href'] if entry.find('link') else '',
                    'updated': entry.find('updated').text if entry.find('updated') else '',
                    'cik': '',
                    'form_type': '',
                    'accession': '',
                }

                # Parse CIK and form type from title
                title = filing['title']
                if ' - ' in title:
                    parts = title.split(' - ')
                    filing['form_type'] = parts[0].strip()

                # Extract accession number from link
                link = filing['link']
                if '/Archives/edgar/data/' in link:
                    match = re.search(r'/data/(\d+)/(\d{10}-\d{2}-\d{6})', link)
                    if match:
                        filing['cik'] = match.group(1)
                        filing['accession'] = match.group(2)

                results.append(filing)

            return results
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")
            return []

    def get_filing_documents(self, cik: str, accession: str) -> List[Dict]:
        """Get list of documents in a filing."""
        accession_clean = accession.replace('-', '')
        url = f"{Config.SEC_BASE}/Archives/edgar/data/{cik}/{accession_clean}/index.json"

        try:
            response = client.get(url)
            data = response.json()

            docs = []
            for item in data.get('directory', {}).get('item', []):
                docs.append({
                    'name': item.get('name', ''),
                    'type': item.get('type', ''),
                    'size': item.get('size', 0),
                    'url': f"{Config.SEC_BASE}/Archives/edgar/data/{cik}/{accession_clean}/{item.get('name', '')}",
                })
            return docs
        except Exception as e:
            console.print(f"[red]Failed to get filing documents: {e}[/red]")
            return []

    def download_full_index(self, year: int, quarter: int) -> Path:
        """Download EDGAR full index for a quarter."""
        url = f"{Config.EDGAR_FULL_INDEX}/{year}/QTR{quarter}/master.zip"
        dest = Config.EDGAR_DIR / "index" / f"{year}_Q{quarter}_master.zip"

        if dest.exists():
            console.print(f"[yellow]Index already exists: {dest.name}[/yellow]")
            return dest

        console.print(f"Downloading index: {year} Q{quarter}...")
        if client.download_file(url, dest):
            return dest
        return None

    def parse_index(self, index_path: Path) -> pd.DataFrame:
        """Parse EDGAR master index file."""
        try:
            with zipfile.ZipFile(index_path, 'r') as zf:
                with zf.open('master.idx') as f:
                    # Skip header lines
                    lines = f.read().decode('latin-1').split('\n')
                    data_start = 0
                    for i, line in enumerate(lines):
                        if line.startswith('---'):
                            data_start = i + 1
                            break

                    records = []
                    for line in lines[data_start:]:
                        if '|' in line:
                            parts = line.split('|')
                            if len(parts) >= 5:
                                records.append({
                                    'cik': parts[0].strip(),
                                    'company': parts[1].strip(),
                                    'form_type': parts[2].strip(),
                                    'date_filed': parts[3].strip(),
                                    'url': f"{Config.SEC_BASE}/Archives/{parts[4].strip()}",
                                })

                    return pd.DataFrame(records)
        except Exception as e:
            console.print(f"[red]Failed to parse index: {e}[/red]")
            return pd.DataFrame()

edgar = EDGARScraper()

# ============================================================================
# FAILS-TO-DELIVER (FTD) ANALYSIS
# ============================================================================

class FTDAnalyzer:
    """Fails-to-Deliver data downloader and analyzer."""

    def __init__(self):
        self.ftd_urls = self._generate_ftd_urls()

    def _generate_ftd_urls(self) -> List[Dict]:
        """Generate FTD file URLs from 2004 to present."""
        urls = []
        current_year = datetime.now().year
        current_month = datetime.now().month

        # FTD data is released ~30 days after the reporting period
        for year in range(2004, current_year + 1):
            for month in range(1, 13):
                if year == current_year and month >= current_month:
                    break

                # First and second half of month
                for half in [1, 2]:
                    if year < 2017:
                        # Old format
                        filename = f"cnsfails{year}{month:02d}{half}.txt"
                    else:
                        # New format
                        if half == 1:
                            filename = f"cnsfails{year}{month:02d}a.zip"
                        else:
                            filename = f"cnsfails{year}{month:02d}b.zip"

                    urls.append({
                        'year': year,
                        'month': month,
                        'half': half,
                        'url': f"https://www.sec.gov/files/data/fails-deliver-data/{filename}",
                        'filename': filename,
                    })

        return urls

    def download_all(self, start_year: int = 2020) -> int:
        """Download all FTD files from start_year to present."""
        downloaded = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            urls = [u for u in self.ftd_urls if u['year'] >= start_year]
            task = progress.add_task("Downloading FTD data...", total=len(urls))

            for item in urls:
                dest = Config.FTD_DIR / item['filename']
                if not dest.exists():
                    if client.download_file(item['url'], dest):
                        downloaded += 1
                progress.advance(task)

        console.print(f"[green]Downloaded {downloaded} new FTD files[/green]")
        return downloaded

    def search_ftd(self, symbol: str = None, cusip: str = None,
                   min_quantity: int = 0, start_date: str = None) -> pd.DataFrame:
        """Search FTD records by symbol, CUSIP, or quantity threshold."""
        results = []

        for ftd_file in Config.FTD_DIR.glob("*"):
            if ftd_file.suffix not in ['.txt', '.zip', '.csv']:
                continue

            try:
                df = self._read_ftd_file(ftd_file)
                if df.empty:
                    continue

                # Apply filters
                if symbol:
                    mask = df['SYMBOL'].str.upper() == symbol.upper()
                    df = df[mask]

                if cusip:
                    mask = df['CUSIP'].str.contains(cusip, case=False, na=False)
                    df = df[mask]

                if min_quantity > 0:
                    df = df[df['QUANTITY FAIL'] >= min_quantity]

                if start_date:
                    df = df[df['SETTLEMENT DATE'] >= start_date]

                if not df.empty:
                    df['SOURCE_FILE'] = ftd_file.name
                    results.append(df)

            except Exception as e:
                continue

        if results:
            combined = pd.concat(results, ignore_index=True)
            combined = combined.sort_values('SETTLEMENT DATE', ascending=False)
            return combined

        return pd.DataFrame()

    def _read_ftd_file(self, path: Path) -> pd.DataFrame:
        """Read FTD file handling various formats."""
        try:
            df = None
            if path.suffix == '.zip':
                with zipfile.ZipFile(path, 'r') as zf:
                    for name in zf.namelist():
                        if name.endswith('.txt') or name.endswith('.csv'):
                            with zf.open(name) as f:
                                df = pd.read_csv(f, sep='|', encoding='latin-1',
                                                 dtype=str, on_bad_lines='skip')
                                break
            else:
                df = pd.read_csv(path, sep='|', encoding='latin-1',
                                 dtype=str, on_bad_lines='skip')

            if df is not None:
                # Normalize column names (handle variations like 'QUANTITY (FAILS)' vs 'QUANTITY FAIL')
                col_map = {}
                for col in df.columns:
                    if 'QUANTITY' in col.upper() and 'FAIL' in col.upper():
                        col_map[col] = 'QUANTITY FAIL'
                if col_map:
                    df = df.rename(columns=col_map)
                return df
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def analyze_ftd_patterns(self, symbol: str) -> Dict:
        """Analyze FTD patterns for a symbol looking for anomalies."""
        df = self.search_ftd(symbol=symbol)

        if df.empty:
            return {'symbol': symbol, 'error': 'No FTD data found'}

        # Convert quantity to numeric
        df['QUANTITY FAIL'] = pd.to_numeric(df['QUANTITY FAIL'], errors='coerce')
        df['PRICE'] = pd.to_numeric(df['PRICE'], errors='coerce')

        analysis = {
            'symbol': symbol,
            'total_records': len(df),
            'date_range': {
                'start': df['SETTLEMENT DATE'].min(),
                'end': df['SETTLEMENT DATE'].max(),
            },
            'quantity_stats': {
                'total': int(df['QUANTITY FAIL'].sum()),
                'mean': float(df['QUANTITY FAIL'].mean()),
                'max': int(df['QUANTITY FAIL'].max()),
                'max_date': df.loc[df['QUANTITY FAIL'].idxmax(), 'SETTLEMENT DATE'],
            },
            'anomalies': [],
        }

        # Detect anomalies (>3 standard deviations)
        mean = df['QUANTITY FAIL'].mean()
        std = df['QUANTITY FAIL'].std()
        if std > 0:
            anomaly_threshold = mean + (3 * std)
            anomalies = df[df['QUANTITY FAIL'] > anomaly_threshold]
            for _, row in anomalies.iterrows():
                analysis['anomalies'].append({
                    'date': row['SETTLEMENT DATE'],
                    'quantity': int(row['QUANTITY FAIL']),
                    'deviation': float((row['QUANTITY FAIL'] - mean) / std),
                })

        return analysis

ftd = FTDAnalyzer()

# ============================================================================
# DERIVATIVES DATA (DTCC)
# ============================================================================

class DerivativesAnalyzer:
    """DTCC derivatives data analyzer for equity and credit swaps."""

    EQUITY_COLUMNS = [
        'Dissemination Identifier', 'Action', 'Execution Timestamp',
        'Underlier ID', 'Notional Amount', 'Notional Currency',
        'Total Notional Quantity', 'Price', 'Price Currency',
    ]

    CREDIT_COLUMNS = [
        'Dissemination Identifier', 'Action', 'Execution Timestamp',
        'Reference Entity', 'Notional Amount', 'Notional Currency',
        'Upfront Payment', 'Fixed Rate', 'Spread',
    ]

    def download_equity_swaps(self, start_date: str, end_date: str,
                               underlying: str = None) -> Path:
        """Download equity derivatives data from DTCC."""
        # DTCC provides daily slice files
        # This is a simplified approach - actual DTCC access requires registration
        dest = Config.DERIVATIVES_DIR / "equity" / f"equity_{start_date}_{end_date}.csv"

        console.print(f"[yellow]Note: DTCC data requires account registration at pddata.dtcc.com[/yellow]")
        console.print(f"[yellow]Download manually and place in: {dest.parent}[/yellow]")

        return dest

    def search_derivatives(self, underlying: str = None,
                          min_notional: float = 0,
                          swap_type: str = 'equity') -> pd.DataFrame:
        """Search downloaded derivatives data."""
        results = []
        search_dir = Config.DERIVATIVES_DIR / swap_type

        if not search_dir.exists():
            return pd.DataFrame()

        for csv_file in search_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file, dtype=str, on_bad_lines='skip')

                if underlying:
                    # Search in various possible column names
                    for col in ['Underlier ID', 'Reference Entity', 'underlying_asset']:
                        if col in df.columns:
                            mask = df[col].str.contains(underlying, case=False, na=False)
                            df = df[mask]
                            break

                if min_notional > 0 and 'Notional Amount' in df.columns:
                    df['Notional Amount'] = pd.to_numeric(df['Notional Amount'], errors='coerce')
                    df = df[df['Notional Amount'] >= min_notional]

                if not df.empty:
                    df['source_file'] = csv_file.name
                    results.append(df)

            except Exception:
                continue

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def analyze_swap_activity(self, underlying: str) -> Dict:
        """Analyze swap activity for potential coordination or manipulation."""
        df = self.search_derivatives(underlying=underlying)

        if df.empty:
            return {'underlying': underlying, 'error': 'No swap data found'}

        # Convert notional to numeric
        if 'Notional Amount' in df.columns:
            df['Notional Amount'] = pd.to_numeric(df['Notional Amount'], errors='coerce')

        analysis = {
            'underlying': underlying,
            'total_transactions': len(df),
            'unique_identifiers': df['Dissemination Identifier'].nunique() if 'Dissemination Identifier' in df.columns else 0,
        }

        if 'Notional Amount' in df.columns:
            analysis['notional_stats'] = {
                'total': float(df['Notional Amount'].sum()),
                'mean': float(df['Notional Amount'].mean()),
                'max': float(df['Notional Amount'].max()),
            }

        # Look for clustering of execution times
        if 'Execution Timestamp' in df.columns:
            df['exec_date'] = pd.to_datetime(df['Execution Timestamp'], errors='coerce').dt.date
            daily_counts = df.groupby('exec_date').size()

            if len(daily_counts) > 0:
                mean_daily = daily_counts.mean()
                std_daily = daily_counts.std()

                if std_daily > 0:
                    spikes = daily_counts[daily_counts > mean_daily + (2 * std_daily)]
                    analysis['activity_spikes'] = [
                        {'date': str(date), 'count': int(count)}
                        for date, count in spikes.items()
                    ]

        return analysis

derivatives = DerivativesAnalyzer()

# ============================================================================
# INSIDER TRADING ANALYSIS
# ============================================================================

class InsiderAnalyzer:
    """Insider trading (Form 3/4/5) analyzer."""

    TRANSACTION_CODES = {
        'P': 'Purchase',
        'S': 'Sale',
        'A': 'Grant/Award',
        'D': 'Disposition to issuer',
        'F': 'Tax withholding',
        'I': 'Discretionary',
        'M': 'Option exercise',
        'C': 'Conversion',
        'E': 'Expiration',
        'G': 'Gift',
        'L': 'Small acquisition',
        'W': 'Will/inheritance',
        'Z': 'Trust',
        'J': '401(k)',
        'K': 'Swap',
        'U': 'Tender',
    }

    def search_insider_filings(self, company: str = None, owner: str = None,
                                form_type: str = '4') -> List[Dict]:
        """Search for insider trading filings."""
        params = {
            'action': 'getcompany',
            'type': form_type,
            'dateb': '',
            'owner': 'include',
            'count': 100,
            'output': 'atom',
        }

        if company:
            if company.isdigit():
                params['CIK'] = company
            else:
                params['company'] = company

        if owner:
            params['owner'] = owner

        try:
            response = client.get(Config.EDGAR_BASE, params=params)
            soup = BeautifulSoup(response.content, 'lxml-xml')

            results = []
            for entry in soup.find_all('entry'):
                filing = {
                    'title': entry.find('title').text if entry.find('title') else '',
                    'link': entry.find('link')['href'] if entry.find('link') else '',
                    'updated': entry.find('updated').text if entry.find('updated') else '',
                    'summary': entry.find('summary').text if entry.find('summary') else '',
                }
                results.append(filing)

            return results
        except Exception as e:
            console.print(f"[red]Search failed: {e}[/red]")
            return []

    def parse_form4(self, url: str) -> Dict:
        """Parse Form 4 XML to extract transaction details."""
        try:
            # Convert HTML URL to XML
            xml_url = url.replace('-index.htm', '.txt')
            if '/000' in xml_url:
                # Try to find the actual XML file
                response = client.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')

                # Look for XML file link
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    if href.endswith('.xml') and 'form4' in href.lower():
                        xml_url = f"{Config.SEC_BASE}{href}"
                        break

            response = client.get(xml_url)
            soup = BeautifulSoup(response.content, 'lxml-xml')

            # Extract issuer info
            issuer = soup.find('issuer')
            owner = soup.find('reportingOwner')

            transactions = []
            for txn in soup.find_all('nonDerivativeTransaction'):
                trans = {
                    'security': txn.find('securityTitle').find('value').text if txn.find('securityTitle') else '',
                    'date': txn.find('transactionDate').find('value').text if txn.find('transactionDate') else '',
                    'code': txn.find('transactionCode').text if txn.find('transactionCode') else '',
                    'shares': '',
                    'price': '',
                    'acquired_disposed': '',
                }

                amounts = txn.find('transactionAmounts')
                if amounts:
                    shares = amounts.find('transactionShares')
                    if shares:
                        trans['shares'] = shares.find('value').text if shares.find('value') else ''

                    price = amounts.find('transactionPricePerShare')
                    if price:
                        trans['price'] = price.find('value').text if price.find('value') else ''

                    ad = amounts.find('transactionAcquiredDisposedCode')
                    if ad:
                        trans['acquired_disposed'] = ad.find('value').text if ad.find('value') else ''

                transactions.append(trans)

            return {
                'issuer_cik': issuer.find('issuerCik').text if issuer and issuer.find('issuerCik') else '',
                'issuer_name': issuer.find('issuerName').text if issuer and issuer.find('issuerName') else '',
                'owner_cik': owner.find('rptOwnerCik').text if owner and owner.find('rptOwnerCik') else '',
                'owner_name': owner.find('rptOwnerName').text if owner and owner.find('rptOwnerName') else '',
                'transactions': transactions,
            }
        except Exception as e:
            return {'error': str(e)}

    def detect_insider_patterns(self, company_cik: str) -> Dict:
        """Detect suspicious insider trading patterns."""
        filings = self.search_insider_filings(company=company_cik)

        analysis = {
            'company_cik': company_cik,
            'total_filings': len(filings),
            'patterns': [],
            'red_flags': [],
        }

        # Track transactions by owner and date
        transactions_by_owner = defaultdict(list)
        transactions_by_date = defaultdict(list)

        for filing in filings[:50]:  # Analyze recent 50
            parsed = self.parse_form4(filing['link'])
            if 'error' in parsed:
                continue

            owner = parsed.get('owner_name', 'Unknown')
            for txn in parsed.get('transactions', []):
                transactions_by_owner[owner].append(txn)
                if txn.get('date'):
                    transactions_by_date[txn['date']].append({
                        'owner': owner,
                        **txn
                    })

        # Look for coordinated selling
        for date, txns in transactions_by_date.items():
            sells = [t for t in txns if t.get('acquired_disposed') == 'D']
            if len(sells) >= 3:
                analysis['red_flags'].append({
                    'type': 'coordinated_selling',
                    'date': date,
                    'count': len(sells),
                    'sellers': [t['owner'] for t in sells],
                })

        # Look for large individual sales
        for owner, txns in transactions_by_owner.items():
            total_sold = sum(
                float(t.get('shares', 0) or 0)
                for t in txns
                if t.get('acquired_disposed') == 'D'
            )
            if total_sold > 100000:
                analysis['patterns'].append({
                    'type': 'large_sales',
                    'owner': owner,
                    'total_shares': total_sold,
                })

        return analysis

insider = InsiderAnalyzer()

# ============================================================================
# TRUTH FINDER - CROSS-REFERENCE ANALYSIS
# ============================================================================

class TruthFinder:
    """Cross-reference analyzer for uncovering hidden connections and patterns."""

    def __init__(self):
        self.red_flags = []

    def full_analysis(self, symbol: str, cik: str = None) -> Dict:
        """Comprehensive analysis of a security across all data sources."""
        console.print(Panel(f"[bold]TRUTH FINDER ANALYSIS: {symbol}[/bold]",
                           title="Gamecock", border_style="blue"))

        results = {
            'symbol': symbol,
            'cik': cik,
            'timestamp': datetime.now().isoformat(),
            'ftd_analysis': {},
            'derivative_analysis': {},
            'insider_analysis': {},
            'cross_references': [],
            'red_flags': [],
            'truth_score': 0,  # 0-100, higher = more suspicious
        }

        # 1. FTD Analysis
        console.print("\n[bold cyan]1. Analyzing Fails-to-Deliver data...[/bold cyan]")
        results['ftd_analysis'] = ftd.analyze_ftd_patterns(symbol)

        if results['ftd_analysis'].get('anomalies'):
            results['red_flags'].append({
                'source': 'FTD',
                'type': 'anomalous_fails',
                'severity': 'HIGH',
                'details': f"{len(results['ftd_analysis']['anomalies'])} anomalous FTD spikes detected",
            })
            results['truth_score'] += 20

        # 2. Derivatives Analysis
        console.print("[bold cyan]2. Analyzing derivatives/swap data...[/bold cyan]")
        results['derivative_analysis'] = derivatives.analyze_swap_activity(symbol)

        if results['derivative_analysis'].get('activity_spikes'):
            results['red_flags'].append({
                'source': 'Derivatives',
                'type': 'coordinated_swaps',
                'severity': 'HIGH',
                'details': f"{len(results['derivative_analysis']['activity_spikes'])} swap activity spikes",
            })
            results['truth_score'] += 25

        # 3. Insider Analysis
        if cik:
            console.print("[bold cyan]3. Analyzing insider trading patterns...[/bold cyan]")
            results['insider_analysis'] = insider.detect_insider_patterns(cik)

            if results['insider_analysis'].get('red_flags'):
                for flag in results['insider_analysis']['red_flags']:
                    results['red_flags'].append({
                        'source': 'Insider',
                        'type': flag['type'],
                        'severity': 'MEDIUM',
                        'details': f"{flag.get('count', 0)} coordinated transactions on {flag.get('date', 'unknown')}",
                    })
                    results['truth_score'] += 15

        # 4. Cross-reference FTD spikes with insider selling
        console.print("[bold cyan]4. Cross-referencing data sources...[/bold cyan]")

        ftd_anomaly_dates = set()
        for anomaly in results['ftd_analysis'].get('anomalies', []):
            if anomaly.get('date'):
                ftd_anomaly_dates.add(anomaly['date'][:10])  # Date only

        insider_flags = results['insider_analysis'].get('red_flags', [])
        for flag in insider_flags:
            if flag.get('date') in ftd_anomaly_dates:
                results['cross_references'].append({
                    'type': 'ftd_insider_correlation',
                    'severity': 'CRITICAL',
                    'date': flag['date'],
                    'details': 'FTD spike coincides with coordinated insider selling',
                })
                results['truth_score'] += 30

        # Cap score at 100
        results['truth_score'] = min(100, results['truth_score'])

        return results

    def display_analysis(self, results: Dict):
        """Display analysis results in a formatted manner."""
        console.print("\n")

        # Summary
        score = results['truth_score']
        if score >= 70:
            score_color = "red"
            verdict = "HIGH SUSPICION"
        elif score >= 40:
            score_color = "yellow"
            verdict = "MODERATE CONCERN"
        else:
            score_color = "green"
            verdict = "NORMAL ACTIVITY"

        console.print(Panel(
            f"[bold {score_color}]SUSPICION SCORE: {score}/100 - {verdict}[/bold {score_color}]",
            title=f"Analysis: {results['symbol']}",
            border_style=score_color
        ))

        # Red Flags
        if results['red_flags']:
            console.print("\n[bold red]RED FLAGS DETECTED:[/bold red]")
            flag_table = Table(show_header=True, header_style="bold red")
            flag_table.add_column("Source")
            flag_table.add_column("Type")
            flag_table.add_column("Severity")
            flag_table.add_column("Details")

            for flag in results['red_flags']:
                severity_color = {"CRITICAL": "red", "HIGH": "yellow", "MEDIUM": "cyan"}.get(flag['severity'], "white")
                flag_table.add_row(
                    flag['source'],
                    flag['type'],
                    f"[{severity_color}]{flag['severity']}[/{severity_color}]",
                    flag['details']
                )

            console.print(flag_table)

        # Cross References
        if results['cross_references']:
            console.print("\n[bold magenta]CROSS-REFERENCE FINDINGS:[/bold magenta]")
            for ref in results['cross_references']:
                console.print(f"  [bold]{ref['type']}[/bold] ({ref['severity']})")
                console.print(f"    Date: {ref['date']}")
                console.print(f"    {ref['details']}")

        # FTD Summary
        ftd_data = results.get('ftd_analysis', {})
        if ftd_data.get('total_records'):
            console.print(f"\n[bold cyan]FTD DATA:[/bold cyan]")
            console.print(f"  Total Records: {ftd_data['total_records']:,}")
            console.print(f"  Date Range: {ftd_data['date_range']['start']} to {ftd_data['date_range']['end']}")
            console.print(f"  Total Failed Shares: {ftd_data['quantity_stats']['total']:,}")
            console.print(f"  Max Single Day: {ftd_data['quantity_stats']['max']:,} on {ftd_data['quantity_stats']['max_date']}")

        console.print("\n")

truth_finder = TruthFinder()

# ============================================================================
# INTERACTIVE CLI
# ============================================================================

def show_menu():
    """Display main menu."""
    console.print("\n")
    console.print(Panel(
        "[bold]GAMECOCK - SEC Filing Truth Finder[/bold]\n\n"
        "Uncover patterns and anomalies in SEC filings",
        title="Main Menu",
        border_style="blue"
    ))

    menu_table = Table(show_header=False, box=None)
    menu_table.add_column("Option", style="cyan", width=6)
    menu_table.add_column("Description")

    menu_table.add_row("1", "Search SEC EDGAR filings")
    menu_table.add_row("2", "Download FTD data")
    menu_table.add_row("3", "Search FTD records")
    menu_table.add_row("4", "Analyze insider trading")
    menu_table.add_row("5", "Search derivatives/swaps")
    menu_table.add_row("6", "[bold]TRUTH FINDER[/bold] - Full analysis")
    menu_table.add_row("7", "Download EDGAR index")
    menu_table.add_row("8", "Form type reference")
    menu_table.add_row("", "")
    menu_table.add_row("q", "Quit")

    console.print(menu_table)
    console.print("")

def search_edgar_interactive():
    """Interactive EDGAR search."""
    query = Prompt.ask("Enter company name or CIK")
    form_type = Prompt.ask("Form type (leave blank for all)", default="")

    results = edgar.search_company(query, form_type if form_type else None)

    if not results:
        console.print("[yellow]No results found[/yellow]")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", width=4)
    table.add_column("Form")
    table.add_column("Date")
    table.add_column("Title", max_width=50)

    for i, r in enumerate(results[:20], 1):
        table.add_row(
            str(i),
            r['form_type'],
            r['updated'][:10] if r['updated'] else '',
            r['title'][:50]
        )

    console.print(table)

def search_ftd_interactive():
    """Interactive FTD search."""
    symbol = Prompt.ask("Enter stock symbol (e.g., GME, AMC)")
    min_qty = Prompt.ask("Minimum fail quantity", default="10000")

    df = ftd.search_ftd(symbol=symbol, min_quantity=int(min_qty))

    if df.empty:
        console.print("[yellow]No FTD records found[/yellow]")
        return

    console.print(f"\n[green]Found {len(df)} records[/green]\n")

    # Show top 20
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Date")
    table.add_column("Symbol")
    table.add_column("CUSIP")
    table.add_column("Quantity", justify="right")
    table.add_column("Price", justify="right")

    for _, row in df.head(20).iterrows():
        table.add_row(
            str(row.get('SETTLEMENT DATE', '')),
            str(row.get('SYMBOL', '')),
            str(row.get('CUSIP', '')),
            f"{int(float(row.get('QUANTITY FAIL', 0))):,}",
            str(row.get('PRICE', ''))
        )

    console.print(table)

    # Offer analysis
    if Confirm.ask("\nRun anomaly analysis?"):
        analysis = ftd.analyze_ftd_patterns(symbol)
        console.print(f"\n[bold]Analysis for {symbol}:[/bold]")
        console.print(f"Total FTD records: {analysis.get('total_records', 0):,}")
        console.print(f"Total shares failed: {analysis.get('quantity_stats', {}).get('total', 0):,}")

        if analysis.get('anomalies'):
            console.print(f"\n[red]Anomalies detected: {len(analysis['anomalies'])}[/red]")
            for a in analysis['anomalies'][:5]:
                console.print(f"  {a['date']}: {a['quantity']:,} shares ({a['deviation']:.1f} std devs)")

def analyze_insider_interactive():
    """Interactive insider trading analysis."""
    company = Prompt.ask("Enter company name or CIK")

    console.print("\n[cyan]Searching insider filings...[/cyan]")
    filings = insider.search_insider_filings(company=company)

    if not filings:
        console.print("[yellow]No insider filings found[/yellow]")
        return

    console.print(f"[green]Found {len(filings)} Form 4 filings[/green]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", width=4)
    table.add_column("Date")
    table.add_column("Title", max_width=60)

    for i, f in enumerate(filings[:15], 1):
        table.add_row(
            str(i),
            f['updated'][:10] if f['updated'] else '',
            f['title'][:60]
        )

    console.print(table)

    if Confirm.ask("\nRun pattern detection?"):
        cik = Prompt.ask("Enter CIK for pattern analysis")
        analysis = insider.detect_insider_patterns(cik)

        if analysis.get('red_flags'):
            console.print(f"\n[red]RED FLAGS DETECTED:[/red]")
            for flag in analysis['red_flags']:
                console.print(f"  [{flag['type']}] {flag.get('date', 'N/A')}: {flag.get('count', 0)} transactions")
                if flag.get('sellers'):
                    for seller in flag['sellers'][:5]:
                        console.print(f"    - {seller}")

def truth_finder_interactive():
    """Run full Truth Finder analysis."""
    symbol = Prompt.ask("Enter stock symbol")
    cik = Prompt.ask("Enter CIK (optional, for insider analysis)", default="")

    results = truth_finder.full_analysis(symbol, cik if cik else None)
    truth_finder.display_analysis(results)

    # Save results
    if Confirm.ask("Save analysis to file?"):
        output_path = Config.ANALYSIS_DIR / f"{symbol}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        console.print(f"[green]Saved to: {output_path}[/green]")

def show_form_reference():
    """Display SEC form type reference."""
    console.print("\n[bold]SEC Form Types Reference[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Form")
    table.add_column("Description")

    for form, desc in sorted(edgar.FORM_TYPES.items()):
        table.add_row(form, desc)

    console.print(table)

def main():
    """Main entry point."""
    console.print(Panel(
        "[bold blue]GAMECOCK[/bold blue]\n"
        "[dim]SEC Filing Truth Finder[/dim]\n\n"
        "An enhanced tool for uncovering patterns and truth\n"
        "within SEC filings, FTD data, and derivatives.\n\n"
        f"Data directory: {Config.BASE_DIR}",
        title="Welcome",
        border_style="blue"
    ))

    while True:
        show_menu()
        choice = Prompt.ask("Select option", default="q")

        try:
            if choice == '1':
                search_edgar_interactive()
            elif choice == '2':
                year = Prompt.ask("Start year for FTD download", default="2020")
                ftd.download_all(int(year))
            elif choice == '3':
                search_ftd_interactive()
            elif choice == '4':
                analyze_insider_interactive()
            elif choice == '5':
                symbol = Prompt.ask("Enter underlying symbol")
                results = derivatives.search_derivatives(underlying=symbol)
                if results.empty:
                    console.print("[yellow]No derivative data found. Download data from DTCC first.[/yellow]")
                else:
                    console.print(results.head(20).to_string())
            elif choice == '6':
                truth_finder_interactive()
            elif choice == '7':
                year = Prompt.ask("Year", default=str(datetime.now().year))
                quarter = Prompt.ask("Quarter (1-4)", default="1")
                edgar.download_full_index(int(year), int(quarter))
            elif choice == '8':
                show_form_reference()
            elif choice.lower() == 'q':
                console.print("[blue]Goodbye![/blue]")
                break
            else:
                console.print("[yellow]Invalid option[/yellow]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

if __name__ == "__main__":
    main()
