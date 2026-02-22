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
# GROOT uses a different timing output format
GROOT_TIMER_PATTERN = re.compile(r"\[KNN_MST_DFS\]\s+Reordering time \(ms\):\s+([0-9.]+)")
# Fallback: SPARTA binary prints "timer: VALUE" in microseconds
SPARTA_TIMER_PATTERN = re.compile(r"^timer:\s+([0-9.eE+\-]+)", re.MULTILINE)

# Known perm job tags (not ANALYSIS_ and not SPMM)
PERM_TAGS = {
    'SB_amd', 'SB_degree', 'SB_gray', 'SB_rcm', 'SB_metis',
    'SB_rabbit', 'SB_patoh', 'SB_slashburn',
    'GROOT_reorder', 'SPARTA_reorder', 'TCA_reorder', 'random1D', 'random2D',
}

# Random-pipeline perm tags: "<algo>_RANDOM" (from perms_random.yaml)
RANDOM_PERM_TAGS = {f'{t}_RANDOM' for t in PERM_TAGS if t not in ('random1D', 'random2D')}

def dedup_latest(df, key_cols):
    """Drop duplicate experiments, keeping only the row with the highest job_id."""
    before = len(df)
    df = df.sort_values('job_id', ascending=False).drop_duplicates(subset=key_cols, keep='first')
    df = df.sort_values('job_id').reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} duplicate rows (kept latest per {key_cols})", file=sys.stderr)
    return df


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

# Canonical label aliases: map non-standard timer labels to standard ones.
# e.g. DTC prints <Timer>[SpMM] instead of <Timer>[operation].
TIMER_LABEL_ALIASES = {
    "spmm": "operation",
}


def parse_timers(stdout):
    """Extract all timers from stdout."""
    if not stdout:
        return {}

    timers = {}
    
    # Remove ANSI color codes only if needed
    clean_stdout = ANSI_ESCAPE.sub('', stdout) if '\x1b' in stdout or '\x1B' in stdout else stdout
    
    # Look for lines like: <Timer>[label] 123.456 ms
    for match in TIMER_PATTERN.finditer(clean_stdout):
        label = match.group(1).lower()
        label = TIMER_LABEL_ALIASES.get(label, label)
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

        job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))

        row = {
            'job_id': job_id,
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


def parse_one_perm_job(job):
    """Parse a single permutation-generation job. Returns a dict row or None."""
    try:
        mtx_path = safe_get_var(job, 'mtx', '')
        matrix_name = get_matrix_name(mtx_path)
        tag = job.tag

        stdout = job.get_stdout()
        if stdout is None:
            return None

        # Clean ANSI codes
        clean = ANSI_ESCAPE.sub('', stdout) if ('\x1b' in stdout or '\x1B' in stdout) else stdout

        time_reordering_ms = None
        time_loading_ms = None

        # Try standard <Timer> format (SPARSEBASE tools)
        for match in TIMER_PATTERN.finditer(clean):
            label = match.group(1)
            value = float(match.group(2))
            if label == 'reordering':
                time_reordering_ms = value
            elif label == 'loading':
                time_loading_ms = value

        # Try GROOT format
        if time_reordering_ms is None:
            groot_match = GROOT_TIMER_PATTERN.search(clean)
            if groot_match:
                time_reordering_ms = float(groot_match.group(1))

        # Fallback: SPARTA binary raw timer (microseconds)
        # The binary prints "timer: VALUE" (us); reorder.py forwards it to stdout.
        # Can be removed once all SPARTA jobs use an updated reorder.py.
        if time_reordering_ms is None:
            sparta_match = SPARTA_TIMER_PATTERN.search(clean)
            if sparta_match:
                time_reordering_ms = float(sparta_match.group(1)) / 1000.0

        # Skip jobs that didn't produce any timing data
        if time_reordering_ms is None:
            return None

        job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))

        return {
            'job_id': job_id,
            'tag': tag,
            'matrix': matrix_name,
            'perm': tag,
            'time_reordering_ms': time_reordering_ms,
            'time_loading_ms': time_loading_ms,
        }

    except Exception as e:
        job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))
        print(f"Error parsing perm job {job_id}: {e}", file=sys.stderr)
        return None


def parse_one_random_perm_job(job):
    """Parse a random-pipeline perm job, stripping the _RANDOM suffix from perm/tag."""
    row = parse_one_perm_job(job)
    if row is None:
        return None
    for key in ('perm', 'tag'):
        if row[key].endswith('_RANDOM'):
            row[key] = row[key][:-len('_RANDOM')]
    return row


def parse_one_random_operation_job(job):
    """Parse a random-pipeline operation job, stripping _RANDOM from perm/tag/algo."""
    row = parse_one_operation_job(job)
    if row is None:
        return None
    for key in ('perm', 'tag', 'algo'):
        if isinstance(row.get(key), str) and row[key].endswith('_RANDOM'):
            row[key] = row[key][:-len('_RANDOM')]
    return row


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
    random_analysis_jobs = []
    op_jobs = []
    random_op_jobs = []
    perm_jobs = []
    random_perm_jobs = []
    for j in all_jobs:
        tag = j.tag or ""
        if tag.startswith("ANALYSIS_RANDOM_"):
            random_analysis_jobs.append(j)
        elif tag.startswith("ANALYSIS_"):
            analysis_jobs.append(j)
        elif "SPMM" in tag and "RANDOM" in tag:
            random_op_jobs.append(j)
        elif "SPMM" in tag:
            op_jobs.append(j)
        elif tag in RANDOM_PERM_TAGS:
            random_perm_jobs.append(j)
        elif tag in PERM_TAGS:
            perm_jobs.append(j)

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
        df_analysis = dedup_latest(df_analysis, ['matrix', 'perm', 'perm_type'])
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
        df_op = dedup_latest(df_op, ['matrix', 'perm', 'perm_type', 'algo', 'block_size', 'n_cols'])
        out_file = out_dir / "results_operations.csv"
        df_op.to_csv(out_file, index=False)
        print(f"Exported {len(df_op)} operation rows to {out_file}")
    else:
        print("No operation results found.")

    # --- 3. Process Permutation Jobs (parallel) ---
    print(f"Found {len(perm_jobs)} permutation jobs.", file=sys.stderr)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(
            pool.map(parse_one_perm_job, perm_jobs, chunksize=256),
            total=len(perm_jobs),
            desc="Parsing Perm Jobs"
        ))
    perm_results = [r for r in results if r is not None]
    t_perms = time.perf_counter() - t0
    print(f"Perm parsing: {t_perms:.1f}s", file=sys.stderr)

    # Export Reordering CSV
    if perm_results:
        df_perm = pd.DataFrame(perm_results)
        df_perm = dedup_latest(df_perm, ['matrix', 'perm'])
        out_file = out_dir / "results_reordering.csv"
        df_perm.to_csv(out_file, index=False)
        print(f"Exported {len(df_perm)} reordering rows to {out_file}")
    else:
        print("No permutation timing results found.")

    # --- 4. Process Random-Pipeline Analysis Jobs (parallel) ---
    print(f"Found {len(random_analysis_jobs)} random-pipeline analysis jobs.", file=sys.stderr)

    t0 = time.perf_counter()
    if random_analysis_jobs:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(
                pool.map(parse_one_analysis_job, random_analysis_jobs, chunksize=256),
                total=len(random_analysis_jobs),
                desc="Parsing Random Analysis Jobs"
            ))
        random_analysis_results = [r for r in results if r is not None]
    else:
        random_analysis_results = []
    t_random_analysis = time.perf_counter() - t0
    print(f"Random analysis parsing: {t_random_analysis:.1f}s", file=sys.stderr)

    # Include random1D SYMMETRIC baseline from the main analysis as the
    # "no reorder" reference point for the random experiment.
    # Re-label perm='None' so it matches the NO_REORDER convention.
    if analysis_results:
        for r in analysis_results:
            if r.get('perm') == 'random1D' and r.get('perm_type') == 'SYMMETRIC':
                baseline = dict(r)
                baseline['perm'] = 'None'
                random_analysis_results.append(baseline)

    if random_analysis_results:
        df_random_analysis = pd.DataFrame(random_analysis_results)
        df_random_analysis = dedup_latest(df_random_analysis, ['matrix', 'perm', 'perm_type'])
        out_file = out_dir / "results_analysis_random.csv"
        df_random_analysis.to_csv(out_file, index=False)
        print(f"Exported {len(df_random_analysis)} random analysis rows to {out_file}")
    else:
        print("No random-pipeline analysis results found.")

    # --- 5. Process Random-Pipeline Operation Jobs (parallel) ---
    print(f"Found {len(random_op_jobs)} random-pipeline operation jobs.", file=sys.stderr)

    t0 = time.perf_counter()
    if random_op_jobs:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(
                pool.map(parse_one_random_operation_job, random_op_jobs, chunksize=256),
                total=len(random_op_jobs),
                desc="Parsing Random Operation Jobs"
            ))
        random_op_results = [r for r in results if r is not None]
    else:
        random_op_results = []
    t_random_ops = time.perf_counter() - t0
    print(f"Random operations parsing: {t_random_ops:.1f}s", file=sys.stderr)

    # Include random1D SYMMETRIC baseline from the main operations as the
    # "no reorder" reference point for the random experiment.
    if op_results:
        for r in op_results:
            if r.get('perm') == 'random1D' and r.get('perm_type') == 'SYMMETRIC':
                baseline = dict(r)
                baseline['perm'] = 'None'
                random_op_results.append(baseline)

    if random_op_results:
        df_random_op = pd.DataFrame(random_op_results)
        df_random_op = dedup_latest(df_random_op, ['matrix', 'perm', 'perm_type', 'algo', 'block_size', 'n_cols'])
        out_file = out_dir / "results_operations_random.csv"
        df_random_op.to_csv(out_file, index=False)
        print(f"Exported {len(df_random_op)} random operation rows to {out_file}")
    else:
        print("No random-pipeline operation results found.")

    # --- 6. Process Random-Pipeline Permutation Jobs (parallel) ---
    print(f"Found {len(random_perm_jobs)} random-pipeline permutation jobs.", file=sys.stderr)

    t0 = time.perf_counter()
    if random_perm_jobs:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(
                pool.map(parse_one_random_perm_job, random_perm_jobs, chunksize=256),
                total=len(random_perm_jobs),
                desc="Parsing Random Perm Jobs"
            ))
        random_perm_results = [r for r in results if r is not None]
    else:
        random_perm_results = []
    t_random_perms = time.perf_counter() - t0
    print(f"Random perm parsing: {t_random_perms:.1f}s", file=sys.stderr)

    if random_perm_results:
        df_random_perm = pd.DataFrame(random_perm_results)
        df_random_perm = dedup_latest(df_random_perm, ['matrix', 'perm'])
        out_file = out_dir / "results_reordering_random.csv"
        df_random_perm.to_csv(out_file, index=False)
        print(f"Exported {len(df_random_perm)} random reordering rows to {out_file}")
    else:
        print("No random-pipeline permutation timing results found.")

    t_total = time.perf_counter() - t_total_start
    print(f"\nTotal time: {t_total:.1f}s  (fetch: {t_fetch:.1f}s, analysis: {t_analysis:.1f}s, "
          f"operations: {t_ops:.1f}s, perms: {t_perms:.1f}s, "
          f"random_analysis: {t_random_analysis:.1f}s, random_ops: {t_random_ops:.1f}s, "
          f"random_perms: {t_random_perms:.1f}s)",
          file=sys.stderr)

if __name__ == "__main__":
    main()
