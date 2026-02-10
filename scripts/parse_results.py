import pandas as pd
import json
import re
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from sbatchman import jobs_list

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Default configuration
DEFAULT_N_COLS = 32
DEFAULT_BLOCK_SIZE = 0

# Pre-compile regex patterns for performance
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
TIMER_PATTERN = re.compile(r"<Timer>\[(.*?)\]\s+([0-9.]+)\s+ms")

def safe_get_var(job, key, default, cast_type=str):
    """Safely extract a variable from job.variables with type casting."""
    # Ensure variables dict exists
    variables = getattr(job, 'variables', {}) or {}
    value = variables.get(key)
    
    if value is None:
        return default
    try:
        return cast_type(value)
    except (ValueError, TypeError):
        return default

def get_matrix_name(path):
    """Extract matrix filename from path."""
    return Path(path).name

def parse_timers(stdout):
    """Extract all timers from stdout."""
    if not stdout:
        return {}

    timers = {}
    
    # Remove ANSI color codes only if needed
    clean_stdout = ANSI_ESCAPE.sub('', stdout) if '\x1b' in stdout or '\x1B' in stdout else stdout
    
    # Look for lines like: <Timer>[label] 123.456 ms
    for match in TIMER_PATTERN.finditer(clean_stdout):
        label = match.group(1)
        value = float(match.group(2))
        timers[f"time_{label}_ms"] = value
    return timers

# Number of threads for parallel I/O (tunable via --workers)
DEFAULT_WORKERS = 32


def _get_perm_type(tag):
    """Determine perm_type from a job tag string."""
    if 'ROW' in tag:
        return 'ROW'
    elif 'SYMMETRIC' in tag:
        return 'SYMMETRIC'
    elif 'ASYMMETRIC' in tag:
        return 'ASYMMETRIC'
    elif 'NO_REORDER' in tag:
        return 'ROW'
    return 'UNKNOWN'


def parse_one_analysis_job(job):
    """Parse a single analysis job. Returns a dict row or None."""
    try:
        mtx_path = safe_get_var(job, 'mtx', '')
        matrix_name = get_matrix_name(mtx_path)
        perm = safe_get_var(job, 'perm', 'None')
        perm_type = _get_perm_type(job.tag)

        stdout = job.get_stdout()
        if stdout is None:
            return None

        start = stdout.find('{')
        end = stdout.rfind('}')

        if start == -1 or end == -1:
            return None

        json_str = stdout[start:end+1]
        data = json.loads(json_str)

        row = {
            'matrix': matrix_name,
            'perm': perm,
            'perm_type': perm_type,
            'rows': data.get('rows'),
            'cols': data.get('cols'),
            'nnz': data.get('nnz'),
            'density': data.get('density'),
        }

        # Flatten Bandwidth
        for k, v in data.get('bandwidth', {}).items():
            row[k] = v

        # Flatten Locality
        for k, v in data.get('locality', {}).items():
            row[f"locality_{k}"] = v

        # Flatten Block Analysis
        for b in data.get('block_analysis', []):
            bs = b.get('block_size')
            if bs:
                row[f'block_density_{bs}'] = b.get('block_density')
                row[f'nonzero_blocks_{bs}'] = b.get('nonzero_blocks')
                row[f'total_blocks_{bs}'] = b.get('total_blocks')
                row[f'max_blocks_per_row_{bs}'] = b.get('max_blocks_per_row')
                row[f'avg_blocks_per_row_{bs}'] = b.get('avg_blocks_per_row')

        return row

    except Exception as e:
        job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))
        print(f"Error parsing analysis job {job_id}: {e}", file=sys.stderr)
        return None


def parse_one_operation_job(job):
    """Parse a single operation job. Returns a dict row or None."""
    try:
        mtx_path = safe_get_var(job, 'mtx', '')
        matrix_name = get_matrix_name(mtx_path)
        perm = safe_get_var(job, 'perm', 'None')
        tag = job.tag
        perm_type = _get_perm_type(tag)

        block_size = safe_get_var(job, 'block_size', DEFAULT_BLOCK_SIZE, int)
        n_cols = safe_get_var(job, 'n_cols', DEFAULT_N_COLS, int)

        timers = parse_timers(job.get_stdout())
        if not timers:
            return None

        job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))

        return {
            'job_id': job_id,
            'tag': tag,
            'matrix': matrix_name,
            'perm': perm,
            'perm_type': perm_type,
            'algo': tag,
            'block_size': block_size if block_size > 0 else None,
            'n_cols': n_cols,
            **timers
        }

    except Exception as e:
        job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))
        print(f"Error parsing operation job {job_id}: {e}", file=sys.stderr)
        return None


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse job results into CSV files.")
    parser.add_argument("--out-dir", default="results", help="Directory for output CSV files")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel I/O threads (default: {DEFAULT_WORKERS})")
    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers
    t_total_start = time.perf_counter()

    print("Fetching jobs (this may take a while)...", file=sys.stderr)
    
    # Fetch ALL completed jobs
    t0 = time.perf_counter()
    try:
        all_jobs = jobs_list(from_archived=True, status=["COMPLETED"], update_jobs=False)
    except Exception as e:
        print(f"Error fetching jobs: {e}", file=sys.stderr)
        sys.exit(1)
    t_fetch = time.perf_counter() - t0
        
    print(f"Total completed jobs found: {len(all_jobs)} ({t_fetch:.1f}s)", file=sys.stderr)
    
    if len(all_jobs) == 0:
        print("No completed jobs found.", file=sys.stderr)
        sys.exit(0)

    # --- Single-pass job categorization ---
    analysis_jobs = []
    op_jobs = []
    for j in all_jobs:
        tag = j.tag or ""
        if tag.startswith("ANALYSIS_"):
            analysis_jobs.append(j)
        elif "SPMM" in tag:
            op_jobs.append(j)

    # --- 1. Process Analysis Jobs (parallel) ---
    print(f"Found {len(analysis_jobs)} analysis jobs.", file=sys.stderr)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(
            pool.map(parse_one_analysis_job, analysis_jobs, chunksize=256),
            total=len(analysis_jobs),
            desc="Parsing Analysis Jobs"
        ))
    analysis_results = [r for r in results if r is not None]
    t_analysis = time.perf_counter() - t0
    print(f"Analysis parsing: {t_analysis:.1f}s", file=sys.stderr)

    # Export Analysis CSV
    if analysis_results:
        df_analysis = pd.DataFrame(analysis_results)
        out_file = out_dir / "results_analysis.csv"
        df_analysis.to_csv(out_file, index=False)
        print(f"Exported {len(df_analysis)} analysis rows to {out_file}")
    else:
        print("No analysis results found.")

    # --- 2. Process Operation Jobs (parallel) ---
    print(f"Found {len(op_jobs)} operation jobs.", file=sys.stderr)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(
            pool.map(parse_one_operation_job, op_jobs, chunksize=256),
            total=len(op_jobs),
            desc="Parsing Operation Jobs"
        ))
    op_results = [r for r in results if r is not None]
    t_ops = time.perf_counter() - t0
    print(f"Operations parsing: {t_ops:.1f}s", file=sys.stderr)

    # Export Operation CSV
    if op_results:
        df_op = pd.DataFrame(op_results)
        out_file = out_dir / "results_operations.csv"
        df_op.to_csv(out_file, index=False)
        print(f"Exported {len(df_op)} operation rows to {out_file}")
    else:
        print("No operation results found.")

    t_total = time.perf_counter() - t_total_start
    print(f"\nTotal time: {t_total:.1f}s  (fetch: {t_fetch:.1f}s, analysis: {t_analysis:.1f}s, operations: {t_ops:.1f}s)",
          file=sys.stderr)

if __name__ == "__main__":
    main()
