# scripts/

Utilities for parsing experiment results, generating plots, and monitoring jobs.

## Files

### parse_results.py

Aggregates experiment outputs from SbatchMan job archives into CSV files.

**Usage:**
```bash
python scripts/parse_results.py
```

**Outputs:**
- `results/results_analysis.csv` - Matrix structural metrics (~17K rows)
- `results/results_operations.csv` - SpMM benchmark timings (~137K rows)

**Key Functions:**
- `parse_analysis_result()` - Parses JSON output from matrix analysis
- `parse_operation_result()` - Parses timing from `<Timer>[label] X.XXX ms` format
- Reads job outputs from `SbatchMan/experiments/*/jobs/*/output.txt`

### plot.py

Generates correlation and performance plots from parsed results.

**Usage:**
```bash
python scripts/plot.py [--one-per-family] [--n-cols N] [--kernel KERNEL]
```

**Arguments:**
- `--one-per-family`: Select one representative matrix per SuiteSparse family
- `--n-cols`: Filter by dense matrix width (32, 256, 1024)
- `--kernel`: Filter by specific kernel name

**Outputs to:** `plots/n_cols_{N}/{kernel}/`

### plot_utils.py

Shared plotting utilities and style configurations.

### job_check.py

Monitors SLURM job status and generates reports.

**Usage:**
```bash
python scripts/job_check.py
```

**Outputs to:** `reports/job_status_{timestamp}.txt`

### check_missing.py

Identifies missing or failed experiment runs.

## Data Flow

```
SbatchMan/experiments/*/jobs/*/output.txt
            │
            ▼
    parse_results.py
            │
            ▼
    results/*.csv
            │
            ▼
       plot.py
            │
            ▼
      plots/*/
```
