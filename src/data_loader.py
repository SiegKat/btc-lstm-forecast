"""
Binance historical kline data: download, extraction, and loading.

Downloads monthly kline ZIP archives from the Binance Vision public
repository, extracts the CSV files, and loads them into DataFrames.

The URL-building and month-trimming logic mirrors the exact offsets
used in the original Colab notebooks.  Changing any index will
produce incorrect URLs or include months with no data.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError

import pandas as pd

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

# ── URL templates ────────────────────────────────────────────────
_PATHS = {
    "ETHUSDT_5m":  "https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/5m/ETHUSDT-5m-",
    "ETHBTC_5m":   "https://data.binance.vision/data/spot/monthly/klines/ETHBTC/5m/ETHBTC-5m-",
    "ETHUSDC_5m":  "https://data.binance.vision/data/spot/monthly/klines/ETHUSDC/5m/ETHUSDC-5m-",
    "BTCUSDT_5m":  "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-",
    "ETHUSDT_15m": "https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/15m/ETHUSDT-15m-",
    "ETHBTC_15m":  "https://data.binance.vision/data/spot/monthly/klines/ETHBTC/15m/ETHBTC-15m-",
    "ETHUSDC_15m": "https://data.binance.vision/data/spot/monthly/klines/ETHUSDC/15m/ETHUSDC-15m-",
    "BTCUSDT_15m": "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-",
}

YEARS = ["2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024"]
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]


def build_all_urls() -> dict[str, list[str]]:
    """Build the full list of monthly ZIP URLs for every symbol/interval.

    Returns a dict keyed by e.g. ``"ETHUSDT_5m"``, ``"BTCUSDT_15m"``.
    """
    result: dict[str, list[str]] = {}
    for key, base in _PATHS.items():
        urls = []
        for year in YEARS:
            for month in MONTHS:
                urls.append(f"{base}{year}-{month}.zip")
        result[key] = urls
    return result


def trim_empty_months(all_urls: dict[str, list[str]]) -> dict[str, list[str]]:
    """Remove months with no data for each pair.

    The slice offsets correspond to the first and last months with actual
    data on Binance for each symbol.  ETHUSDC has an additional gap
    (months 46-50) that is spliced out.

    **Do not change these indices** without verifying against Binance.
    """
    trimmed: dict[str, list[str]] = {}

    trimmed["ETHUSDT_5m"]  = all_urls["ETHUSDT_5m"][7:-3]
    trimmed["ETHBTC_5m"]   = all_urls["ETHBTC_5m"][6:-3]
    trimmed["BTCUSDT_5m"]  = all_urls["BTCUSDT_5m"][7:-3]
    trimmed["ETHUSDT_15m"] = all_urls["ETHUSDT_15m"][7:-3]
    trimmed["ETHBTC_15m"]  = all_urls["ETHBTC_15m"][6:-3]
    trimmed["BTCUSDT_15m"] = all_urls["BTCUSDT_15m"][7:-3]

    usdc_5m  = all_urls["ETHUSDC_5m"][23:-3]
    usdc_15m = all_urls["ETHUSDC_15m"][23:-3]
    trimmed["ETHUSDC_5m"]  = usdc_5m[:46]  + usdc_5m[51:]
    trimmed["ETHUSDC_15m"] = usdc_15m[:46] + usdc_15m[51:]

    return trimmed


# ── Download & extraction ────────────────────────────────────────

def unzip_data(zip_path: str | Path) -> str:
    """Extract the CSV from a Binance kline ZIP archive.

    Returns the path to the extracted CSV file.
    """
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(zip_path.parent)
    csv_path = zip_path.with_suffix(".csv")
    return str(csv_path)


def download_file(url: str, dest_dir: str | Path) -> Path:
    """Download a single ZIP file.  Returns the local path.

    Skips the download if the file (or its extracted CSV) already exists.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = url.split("/")[-1]
    zip_path = dest_dir / filename
    csv_path = zip_path.with_suffix(".csv")

    if csv_path.exists():
        return csv_path
    if zip_path.exists():
        unzip_data(zip_path)
        return csv_path

    try:
        urlretrieve(url, str(zip_path))
        unzip_data(zip_path)
        return csv_path
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def check_dataset_complete(
    urls: list[str],
    data_dir: str | Path,
) -> tuple[bool, int, int]:
    """Check whether all expected CSV files are already present.

    Returns ``(is_complete, n_present, n_total)``.
    """
    data_dir = Path(data_dir)
    total = len(urls)
    present = 0
    for url in urls:
        filename = url.split("/")[-1]
        csv_name = filename.replace(".zip", ".csv")
        if (data_dir / csv_name).exists():
            present += 1
    return present == total, present, total


def download_pair(
    urls: list[str],
    data_dir: str | Path,
    quiet: bool = False,
) -> list[Path]:
    """Download all ZIP files for a set of URLs and extract them.

    Returns a list of local CSV paths (in URL order).
    """
    data_dir = Path(data_dir)
    csv_paths: list[Path] = []

    for i, url in enumerate(urls):
        try:
            csv_path = download_file(url, data_dir)
            csv_paths.append(csv_path)
        except RuntimeError:
            if not quiet:
                print(f"  [skip] {url.split('/')[-1]}")
            continue

    return csv_paths


def load_csvs(
    csv_paths: list[Path],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate a list of CSV files into a single DataFrame."""
    if columns is None:
        columns = KLINE_COLUMNS

    frames = []
    for p in csv_paths:
        try:
            frames.append(pd.read_csv(p, header=0, names=columns))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


# ── High-level convenience ───────────────────────────────────────

def download_and_load_all(
    data_dir: str | Path = "data",
) -> dict[str, pd.DataFrame]:
    """Download (if needed) and load all four pairs at both intervals.

    Parameters
    ----------
    data_dir : str | Path
        Local directory where ZIP/CSV files are stored.  Files are
        re-used across runs (cache).

    Returns
    -------
    dict mapping e.g. ``"BTCUSDT_15m"`` to a DataFrame.
    """
    all_urls = build_all_urls()
    urls = trim_empty_months(all_urls)

    data: dict[str, pd.DataFrame] = {}
    for key in urls:
        complete, present, total = check_dataset_complete(urls[key], data_dir)
        status = "cached" if complete else f"{present}/{total} cached"
        print(f"  {key:16s}  {total:3d} files  ({status})")

        csv_paths = download_pair(urls[key], data_dir)
        data[key] = load_csvs(csv_paths)
        print(f"  {key:16s}  {len(data[key]):>10,} rows loaded")

    return data


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by open_time, drop metadata columns, cast to numeric."""
    df = df.copy()
    df = df.sort_values(by="open_time").reset_index(drop=True)

    drop_cols = ["close_time", "quote_volume", "count",
                 "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=existing)

    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df
