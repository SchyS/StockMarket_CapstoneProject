from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    'entry_date', 'exit_date', 'strategy', 'pnl_price', 'profitable', 'Date',
    'Close', 'High', 'Low', 'Open', 'Volume'
]


def find_dataset(repo_root: Path) -> Path:
    candidates = sorted(repo_root.rglob('Clean_dataset')) + sorted(repo_root.rglob('*.csv'))
    for path in candidates:
        if path.name == 'Clean_dataset':
            return path
    for path in candidates:
        if 'clean' in path.name.lower() and path.is_file():
            return path
    raise FileNotFoundError('Could not find dataset file. Expected a file named Clean_dataset or a clean CSV.')


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_path = find_dataset(repo_root)
    df = pd.read_csv(dataset_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    numeric_price_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
    for col in numeric_price_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f'Column {col} must be numeric')

    if (df[['Close', 'High', 'Low', 'Open']] <= 0).any().any():
        raise ValueError('Price columns contain zero or negative values')

    if (df['Volume'] < 0).any():
        raise ValueError('Volume contains negative values')

    if not set(df['profitable'].dropna().unique()).issubset({0, 1}):
        raise ValueError("'profitable' must contain only 0/1 values")

    date_na = pd.to_datetime(df['Date'], errors='coerce').isna().sum()
    if date_na > 0:
        raise ValueError(f'Date column contains {date_na} invalid values')

    if df.duplicated().any():
        raise ValueError('Dataset contains fully duplicated rows')

    summary = {
        'dataset_path': str(dataset_path.relative_to(repo_root)),
        'rows': int(df.shape[0]),
        'columns': int(df.shape[1]),
        'strategies': sorted(df['strategy'].dropna().astype(str).unique().tolist()),
        'null_counts': df[REQUIRED_COLUMNS].isna().sum().to_dict(),
        'class_balance': df['profitable'].value_counts(dropna=False).to_dict(),
    }

    out_dir = repo_root / 'artifacts'
    out_dir.mkdir(exist_ok=True)
    (out_dir / 'dataset_validation_summary.json').write_text(json.dumps(summary, indent=2))

    print('Dataset validation passed.')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
