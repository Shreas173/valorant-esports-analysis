Merge and clean Valorant CSV snapshots

Usage (quick test, sample 200 files):

```bash
python3 analysis/merge_and_clean.py --source_dir bronze --out analysis/merged_sample.parquet --sample 200
```

Full merge:

```bash
python3 analysis/merge_and_clean.py --source_dir bronze --out analysis/merged.parquet
```

Install dependencies:

```bash
pip install -r analysis/requirements.txt
```
