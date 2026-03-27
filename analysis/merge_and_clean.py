#!/usr/bin/env python3
"""
Merge and clean Valorant CSV snapshots into a single Parquet/CSV for analysis.
Usage:
  python analysis/merge_and_clean.py --source_dir bronze --out analysis/merged.parquet --sample 200
"""
import argparse
import glob
import os
import pandas as pd
from tqdm import tqdm


def parse_path_metadata(path, source_root):
    # expects paths like .../bronze/event_id=8/region=na/map=split/agent=viper/snapshot_date=2026-02-28/data.csv
    rel = os.path.relpath(path, source_root)
    parts = rel.split(os.sep)
    meta = { }
    for p in parts:
        if p.startswith('event_id='):
            meta['event_id'] = p.split('=',1)[1]
        elif p.startswith('region='):
            meta['region'] = p.split('=',1)[1]
        elif p.startswith('map='):
            meta['map'] = p.split('=',1)[1]
        elif p.startswith('agent='):
            meta['agent'] = p.split('=',1)[1]
        elif p.startswith('snapshot_date='):
            meta['snapshot_date'] = p.split('=',1)[1]
    return meta


def percent_to_float(x):
    if pd.isna(x):
        return None
    if isinstance(x, str):
        s = x.strip()
        if s.endswith('%'):
            try:
                return float(s.strip('%'))/100.0
            except ValueError:
                return None
        try:
            return float(s)
        except ValueError:
            return None
    try:
        return float(x)
    except Exception:
        return None


def parse_clutch_ratio(x):
    # returns (won, played)
    if pd.isna(x):
        return (None, None)
    if isinstance(x, str) and '/' in x:
        a,b = x.split('/',1)
        try:
            return (int(a), int(b))
        except ValueError:
            return (None, None)
    return (None, None)


def smart_convert_df(df):
    # Convert percentage-like columns
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        if sample.shape[0]==0:
            continue
        if sample.str.endswith('%').any():
            df[col] = df[col].apply(percent_to_float)
    # attempt numeric conversions for remaining columns
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
    return df


def process_file(path, source_root):
    try:
        df = pd.read_csv(path)
    except Exception:
        # try with engine python for odd csvs
        df = pd.read_csv(path, engine='python')
    meta = parse_path_metadata(path, source_root)
    for k,v in meta.items():
        df[k] = v
    # clean percent-like fields
    df = smart_convert_df(df)
    # parse clutch ratio if present
    if 'clutches_won_played_ratio' in df.columns:
        won_played = df['clutches_won_played_ratio'].apply(lambda x: parse_clutch_ratio(x))
        df['clutches_won'] = won_played.apply(lambda x: x[0])
        df['clutches_played'] = won_played.apply(lambda x: x[1])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default='bronze')
    parser.add_argument('--out', default='analysis/merged.parquet')
    parser.add_argument('--sample', type=int, default=0, help='If >0, limit to first N files for a quick test')
    parser.add_argument('--format', choices=['parquet','csv'], default='parquet')
    args = parser.parse_args()

    src = args.source_dir
    pattern = os.path.join(src, '**', 'data.csv')
    files = glob.glob(pattern, recursive=True)
    files.sort()
    if args.sample and args.sample > 0:
        files = files[:args.sample]

    if len(files) == 0:
        print('No data.csv files found under', src)
        return

    out_rows = []
    # Process files iteratively to keep memory reasonable
    dfs = []
    for f in tqdm(files, desc='files'):
        try:
            df = process_file(f, src)
            dfs.append(df)
        except Exception as e:
            print('Error processing', f, e)
    if len(dfs) == 0:
        print('No dataframes created.')
        return
    merged = pd.concat(dfs, ignore_index=True, copy=False)

    # Final cleanups - ensure headshot percentage normalized
    if 'headshot_percentage' in merged.columns:
        merged['headshot_percentage'] = merged['headshot_percentage'].apply(percent_to_float)

    # Save
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.format == 'parquet':
        merged.to_parquet(out, index=False)
    else:
        merged.to_csv(out, index=False)

    print('Merged', len(files), 'files ->', out)
    print('Result rows:', merged.shape[0])

if __name__ == '__main__':
    main()
