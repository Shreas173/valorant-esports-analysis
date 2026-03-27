#!/usr/bin/env python3
"""
Stream-merge all data.csv files under bronze into a single CSV without loading everything into memory.
Usage:
  analysis/venv/bin/python3 analysis/stream_merge.py --source_dir bronze --out analysis/merged_full.csv
"""
import argparse
import glob
import os
import pandas as pd
from tqdm import tqdm


def parse_path_metadata(path, source_root):
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
    for col in df.columns:
        sample = df[col].dropna().astype(str)
        if sample.shape[0]==0:
            continue
        if sample.str.endswith('%').any():
            df[col] = df[col].apply(percent_to_float)
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
        df = pd.read_csv(path, engine='python')
    meta = parse_path_metadata(path, source_root)
    for k,v in meta.items():
        df[k] = v
    df = smart_convert_df(df)
    if 'clutches_won_played_ratio' in df.columns:
        won_played = df['clutches_won_played_ratio'].apply(lambda x: parse_clutch_ratio(x))
        df['clutches_won'] = won_played.apply(lambda x: x[0])
        df['clutches_played'] = won_played.apply(lambda x: x[1])
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default='bronze')
    parser.add_argument('--out', default='analysis/merged_full.csv')
    parser.add_argument('--limit', type=int, default=0, help='If >0, limit to first N files')
    args = parser.parse_args()

    src = args.source_dir
    pattern = os.path.join(src, '**', 'data.csv')
    files = glob.glob(pattern, recursive=True)
    files.sort()
    if args.limit and args.limit > 0:
        files = files[:args.limit]

    if len(files) == 0:
        print('No data.csv files found under', src)
        return

    out = args.out
    first = True
    written = 0
    for f in tqdm(files, desc='files'):
        try:
            df = process_file(f, src)
            if df.shape[0] == 0:
                continue
            # Ensure consistent column order: if first write header
            if first:
                df.to_csv(out, index=False, mode='w')
                first = False
            else:
                df.to_csv(out, index=False, header=False, mode='a')
            written += df.shape[0]
        except Exception as e:
            print('Error processing', f, e)
    print('Merged', len(files), 'files ->', out)
    print('Result rows:', written)

if __name__ == '__main__':
    main()
