import pandas as pd
import json
import pickle
import re
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

try:
    import yaml
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    import yaml
    from yaml import SafeLoader

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

# Cache format version — bump when parsed row schema changes
CACHE_VERSION = 4


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


def _categorize_tag(tag):
    """Return the category string for a job tag, or None if uncategorized."""
    if tag.startswith("ANALYSIS_RANDOM_"):
        return 'random_analysis'
    elif tag.startswith("ANALYSIS_"):
        return 'analysis'
    elif ("SPMM" in tag or "BLEST_BFS" in tag) and "RANDOM" in tag:
        return 'random_ops'
    elif "SPMM" in tag or "BLEST_BFS" in tag:
        return 'ops'
    elif tag in RANDOM_PERM_TAGS:
        return 'random_perms'
    elif tag in PERM_TAGS:
        return 'perms'
    return None


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

        # Flatten Access Distances
        for k, v in data.get('access_distances', {}).items():
            row[f"access_dist_{k}"] = v

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


def parse_one_random_analysis_job(job):
    """Parse a random-pipeline analysis job, stripping _RANDOM suffix from perm/tag/algo."""
    row = parse_one_analysis_job(job)
    if row is None:
        return None
    for key in ('perm', 'tag', 'algo'):
        if isinstance(row.get(key), str) and row[key].endswith('_RANDOM'):
            row[key] = row[key][:-len('_RANDOM')]
    return row


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


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_cache(cache_path):
    """Load the parse cache. Returns None if missing or incompatible."""
    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        if cache.get('version') != CACHE_VERSION:
            print("Cache version mismatch, rebuilding.", file=sys.stderr)
            return None
        return cache
    except Exception:
        return None


def _save_cache(cache_path, cache):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(cache_path)


def _scan_job_dirs():
    """
    Fast directory scan (no YAML parsing) returning all job directory paths.
    Returns a set of absolute dir path strings (the timestamp-level dirs that
    contain metadata.yaml / stdout.log).
    """
    from sbatchman.config.project_config import get_experiments_dir, get_archive_dir

    dirs = set()

    def _scan_tag_level(tag_dir_path):
        """Scan timestamp dirs under a tag dir."""
        try:
            with os.scandir(tag_dir_path) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        dirs.add(entry.path)
        except OSError:
            pass

    def _scan_from_root(base, extra_level=False):
        """
        Scan the directory tree under base.
        extra_level=True for archive (archive_name/cluster/config/tag/timestamp)
        extra_level=False for experiments (cluster/config/tag/timestamp)
        """
        if not base.exists():
            return
        try:
            level1_entries = list(os.scandir(base))
        except OSError:
            return
        for l1 in level1_entries:
            if not l1.is_dir(follow_symlinks=False):
                continue
            if extra_level:
                # l1 = archive_name, need one more level before cluster
                try:
                    cluster_entries = list(os.scandir(l1.path))
                except OSError:
                    continue
            else:
                cluster_entries = [l1]

            for cluster_entry in cluster_entries:
                if extra_level and not cluster_entry.is_dir(follow_symlinks=False):
                    continue
                try:
                    with os.scandir(cluster_entry.path) as config_it:
                        config_entries = list(config_it)
                except OSError:
                    continue
                for config_entry in config_entries:
                    if not config_entry.is_dir(follow_symlinks=False):
                        continue
                    try:
                        with os.scandir(config_entry.path) as tag_it:
                            tag_entries = list(tag_it)
                    except OSError:
                        continue
                    for tag_entry in tag_entries:
                        if not tag_entry.is_dir(follow_symlinks=False):
                            continue
                        _scan_tag_level(tag_entry.path)

    _scan_from_root(get_experiments_dir(), extra_level=False)
    _scan_from_root(get_archive_dir(), extra_level=True)
    return dirs


def _load_and_parse_one(job_dir_str):
    """
    Load metadata.yaml from a job dir, check it's COMPLETED, parse stdout,
    and return (category, parsed_row) or None.
    Bypasses sbatchman's jobs_list entirely.
    """
    job_dir = Path(job_dir_str)
    metadata_path = job_dir / "metadata.yaml"

    try:
        with open(metadata_path, 'r') as f:
            meta = yaml.load(f, Loader=SafeLoader)
    except Exception:
        return None

    if not meta or str(meta.get('status', '')) != 'COMPLETED':
        return None

    tag = meta.get('tag', '')
    category = _categorize_tag(tag)
    if category is None:
        return None

    # Build a lightweight object with the fields the parse_one_* functions need
    class _JobProxy:
        pass

    job = _JobProxy()
    job.tag = tag
    job.job_id = meta.get('job_id', 'unknown')
    job.id = job.job_id
    job.variables = meta.get('variables') or {}

    # Read stdout directly
    stdout_path = job_dir / "stdout.log"
    try:
        with open(stdout_path, 'r') as f:
            job._stdout = f.read()
    except Exception:
        job._stdout = None
    job.get_stdout = lambda: job._stdout

    # Parse according to category
    if category in ('analysis', 'random_analysis'):
        row = parse_one_analysis_job(job)
    elif category in ('ops', 'random_ops'):
        row = parse_one_operation_job(job)
    elif category in ('perms', 'random_perms'):
        row = parse_one_perm_job(job)
    else:
        return None

    if row is None:
        return None

    # Strip _RANDOM suffix for random categories
    if category.startswith('random_'):
        for key in ('perm', 'tag', 'algo'):
            if isinstance(row.get(key), str) and row[key].endswith('_RANDOM'):
                row[key] = row[key][:-len('_RANDOM')]

    return (category, row)


# ---------------------------------------------------------------------------
# CSV export helpers (shared between cached and uncached paths)
# ---------------------------------------------------------------------------

CATEGORIES = ['analysis', 'random_analysis', 'ops', 'random_ops', 'perms', 'random_perms']


def _empty_rows():
    return {c: [] for c in CATEGORIES}


def _backfill_job_dirs(rows, all_dirs):
    """
    Populate '_job_dir' on rows produced by the uncached path (which lack it).
    Scans metadata.yaml in each dir to build a job_id -> dir mapping, then
    stamps each row.  Rows that can't be matched get _job_dir = None.
    """
    # Build job_id -> dir mapping
    job_id_to_dir = {}
    for d in all_dirs:
        meta_path = Path(d) / "metadata.yaml"
        try:
            with open(meta_path, 'r') as f:
                meta = yaml.load(f, Loader=SafeLoader)
            jid = meta.get('job_id')
            if jid is not None:
                job_id_to_dir[str(jid)] = d
        except Exception:
            continue

    matched = 0
    for cat in CATEGORIES:
        for row in rows[cat]:
            jid = str(row.get('job_id', ''))
            row['_job_dir'] = job_id_to_dir.get(jid)
            if row['_job_dir'] is not None:
                matched += 1

    total = sum(len(rows[c]) for c in CATEGORIES)
    print(f"  Backfilled _job_dir: {matched}/{total} rows matched", file=sys.stderr)


def _add_random_baselines(rows):
    """Copy random1D SYMMETRIC rows from main into random categories as baselines."""
    # analysis -> random_analysis
    for r in rows['analysis']:
        if r.get('perm') == 'random1D' and r.get('perm_type') == 'SYMMETRIC':
            baseline = dict(r)
            baseline['perm'] = 'None'
            rows['random_analysis'].append(baseline)
    # ops -> random_ops
    for r in rows['ops']:
        if r.get('perm') == 'random1D' and r.get('perm_type') == 'SYMMETRIC':
            baseline = dict(r)
            baseline['perm'] = 'None'
            rows['random_ops'].append(baseline)


def _export_csvs(rows, out_dir):
    """Export all category DataFrames to CSV files."""
    csv_map = {
        'analysis':        ('results_analysis.csv',          ['matrix', 'perm', 'perm_type']),
        'random_analysis': ('results_analysis_random.csv',   ['matrix', 'perm', 'perm_type']),
        'ops':             ('results_operations.csv',        ['matrix', 'perm', 'perm_type', 'algo', 'block_size', 'n_cols']),
        'random_ops':      ('results_operations_random.csv', ['matrix', 'perm', 'perm_type', 'algo', 'block_size', 'n_cols']),
        'perms':           ('results_reordering.csv',        ['matrix', 'perm']),
        'random_perms':    ('results_reordering_random.csv', ['matrix', 'perm']),
    }
    for cat, (filename, dedup_keys) in csv_map.items():
        data = rows[cat]
        if data:
            df = pd.DataFrame(data)
            df = dedup_latest(df, dedup_keys)
            out_file = out_dir / filename
            df.to_csv(out_file, index=False)
            print(f"Exported {len(df)} {cat} rows to {out_file}")
        else:
            print(f"No {cat} results found.")


# ---------------------------------------------------------------------------
# Main: uncached path (first run / --no-cache)
# ---------------------------------------------------------------------------

def _run_uncached(workers):
    """Original approach: fetch all jobs via sbatchman, parse, return rows dict."""
    from sbatchman import jobs_list

    print("Fetching jobs (this may take a while)...", file=sys.stderr)
    t0 = time.perf_counter()
    try:
        all_jobs = jobs_list(from_archived=True, status=["COMPLETED"], update_jobs=False)
    except Exception as e:
        print(f"Error fetching jobs: {e}", file=sys.stderr)
        sys.exit(1)
    t_fetch = time.perf_counter() - t0
    print(f"Total completed jobs found: {len(all_jobs)} ({t_fetch:.1f}s)", file=sys.stderr)

    if not all_jobs:
        print("No completed jobs found.", file=sys.stderr)
        sys.exit(0)

    # Single-pass categorization
    categorized = {c: [] for c in CATEGORIES}
    for j in all_jobs:
        cat = _categorize_tag(j.tag or "")
        if cat:
            categorized[cat].append(j)

    rows = _empty_rows()
    parse_fn = {
        'analysis':        parse_one_analysis_job,
        'random_analysis': parse_one_random_analysis_job,
        'ops':             parse_one_operation_job,
        'random_ops':      parse_one_random_operation_job,
        'perms':           parse_one_perm_job,
        'random_perms':    parse_one_random_perm_job,
    }

    for cat in CATEGORIES:
        jobs = categorized[cat]
        if not jobs:
            continue
        print(f"Parsing {len(jobs)} {cat} jobs...", file=sys.stderr)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(
                pool.map(parse_fn[cat], jobs, chunksize=256),
                total=len(jobs), desc=f"  {cat}"
            ))
        rows[cat] = [r for r in results if r is not None]
        print(f"  {cat}: {time.perf_counter() - t0:.1f}s, {len(rows[cat])} parsed", file=sys.stderr)

    return rows


# ---------------------------------------------------------------------------
# Main: cached path (fast incremental)
# ---------------------------------------------------------------------------

def _run_cached(cache, workers):
    """
    Incremental path: scan directories, diff against cache, only process new
    job dirs, merge with cached rows.
    """
    t0 = time.perf_counter()
    all_dirs = _scan_job_dirs()
    t_scan = time.perf_counter() - t0
    print(f"Directory scan: {len(all_dirs)} job dirs ({t_scan:.1f}s)", file=sys.stderr)

    known_dirs = cache.get('known_dirs', set())
    new_dirs = all_dirs - known_dirs
    removed_dirs = known_dirs - all_dirs

    print(f"  New: {len(new_dirs)}, Removed: {len(removed_dirs)}, Cached: {len(known_dirs)}", file=sys.stderr)

    # Start from cached rows, removing any from deleted dirs
    rows = _empty_rows()
    if removed_dirs:
        removed_set = removed_dirs
        for cat in CATEGORIES:
            rows[cat] = [r for r in cache.get(cat, []) if r.get('_job_dir') not in removed_set]
    else:
        for cat in CATEGORIES:
            rows[cat] = list(cache.get(cat, []))

    # Process new dirs
    if new_dirs:
        new_dirs_list = list(new_dirs)
        print(f"Processing {len(new_dirs_list)} new jobs...", file=sys.stderr)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(
                pool.map(_load_and_parse_one, new_dirs_list, chunksize=256),
                total=len(new_dirs_list), desc="  new jobs"
            ))
        new_count = 0
        for i, result in enumerate(results):
            if result is not None:
                cat, row = result
                row['_job_dir'] = new_dirs_list[i]
                rows[cat].append(row)
                new_count += 1
        t_parse = time.perf_counter() - t0
        print(f"  Parsed {new_count} new rows ({t_parse:.1f}s)", file=sys.stderr)
    else:
        print("No new jobs to process.", file=sys.stderr)

    return rows, all_dirs


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse job results into CSV files.")
    parser.add_argument("--out-dir", default="results", help="Directory for output CSV files")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Number of parallel I/O threads (default: {DEFAULT_WORKERS})")
    parser.add_argument("--no-cache", action="store_true",
                        help="Ignore cache and re-parse everything from scratch")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Store cache in project root so it's reused regardless of --out-dir
    cache_path = Path(__file__).resolve().parent.parent / ".parse_cache.pkl"
    workers = args.workers
    t_total_start = time.perf_counter()

    cache = None
    if not args.no_cache:
        cache = _load_cache(cache_path)

    if cache is not None:
        # --- Fast incremental path ---
        rows, all_dirs = _run_cached(cache, workers)

        # Save cache BEFORE adding baselines (baselines are export-only)
        new_cache = {'version': CACHE_VERSION, 'known_dirs': all_dirs}
        for cat in CATEGORIES:
            new_cache[cat] = rows[cat]
        try:
            _save_cache(cache_path, new_cache)
        except OSError as e:
            print(f"WARNING: failed to save cache: {e}", file=sys.stderr)

        # Add random baselines (export-only, not cached)
        _add_random_baselines(rows)

        # Export CSVs (strip internal _job_dir before export)
        export_rows = _empty_rows()
        for cat in CATEGORIES:
            export_rows[cat] = [{k: v for k, v in r.items() if k != '_job_dir'} for r in rows[cat]]
        _export_csvs(export_rows, out_dir)

    else:
        # --- Full parse path (first run or --no-cache) ---
        rows = _run_uncached(workers)

        # Build and save cache BEFORE adding baselines.
        # Scan dirs to populate known_dirs; rows from uncached path need
        # _job_dir populated via _backfill_job_dirs.
        print("Building cache for next run...", file=sys.stderr)
        t0 = time.perf_counter()
        all_dirs = _scan_job_dirs()
        _backfill_job_dirs(rows, all_dirs)
        new_cache = {'version': CACHE_VERSION, 'known_dirs': all_dirs}
        for cat in CATEGORIES:
            new_cache[cat] = rows[cat]
        try:
            _save_cache(cache_path, new_cache)
            print(f"Cache saved ({time.perf_counter() - t0:.1f}s)", file=sys.stderr)
        except OSError as e:
            print(f"WARNING: failed to save cache: {e}", file=sys.stderr)

        # Add random baselines (export-only, not cached)
        _add_random_baselines(rows)

        # Export CSVs (strip _job_dir before export)
        export_rows = _empty_rows()
        for cat in CATEGORIES:
            export_rows[cat] = [{k: v for k, v in r.items() if k != '_job_dir'} for r in rows[cat]]
        _export_csvs(export_rows, out_dir)

    t_total = time.perf_counter() - t_total_start
    print(f"\nTotal time: {t_total:.1f}s", file=sys.stderr)

if __name__ == "__main__":
    main()
