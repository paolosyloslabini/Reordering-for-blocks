import pandas as pd
import json
import re
import os
import sys
from pathlib import Path
from sbatchman import jobs_list

# Default configuration
DEFAULT_N_COLS = 32
DEFAULT_BLOCK_SIZE = 0

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
    
    # Remove ANSI color codes (robust regex for escape sequences)
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_stdout = ansi_escape.sub('', stdout)
    
    # Look for lines like: <Timer>[label] 123.456 ms
    pattern = r"<Timer>\[(.*?)\]\s+([0-9.]+)\s+ms"
    for match in re.finditer(pattern, clean_stdout):
        label = match.group(1)
        value = float(match.group(2))
        timers[f"time_{label}_ms"] = value
    return timers

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Parse job results into CSV files.")
    parser.add_argument("--out-dir", default="results", help="Directory for output CSV files")
    args = parser.parse_args()

    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching jobs...", file=sys.stderr)
    
    # Fetch ALL completed jobs
    try:
        all_jobs = jobs_list(from_archived=True, status=["COMPLETED"])
    except Exception as e:
        print(f"Error fetching jobs: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Total completed jobs found: {len(all_jobs)}", file=sys.stderr)
    
    if len(all_jobs) == 0:
        print("No completed jobs found.", file=sys.stderr)
        sys.exit(0)

    # --- 1. Process Analysis Jobs ---
    analysis_results = []
    analysis_jobs = [j for j in all_jobs if j.tag and j.tag.startswith("ANALYSIS_")]
    print(f"Found {len(analysis_jobs)} analysis jobs.", file=sys.stderr)
    
    for job in analysis_jobs:
        try:
            # Parse variables
            mtx_path = safe_get_var(job, 'mtx', '')
            matrix_name = get_matrix_name(mtx_path)
            perm = safe_get_var(job, 'perm', 'None')
            
            # Determine perm_type
            tag = job.tag
            if 'ROW' in tag: perm_type = 'ROW'
            elif 'SYMMETRIC' in tag: perm_type = 'SYMMETRIC'
            elif 'ASYMMETRIC' in tag: perm_type = 'ASYMMETRIC'
            elif 'NO_REORDER' in tag: perm_type = 'ROW'
            else: perm_type = 'UNKNOWN'
            
            # Parse JSON output
            stdout = job.get_stdout()
            start = stdout.find('{')
            end = stdout.rfind('}')
            
            if start != -1 and end != -1:
                json_str = stdout[start:end+1]
                data = json.loads(json_str)
                
                # Base row
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
                bw = data.get('bandwidth', {})
                for k, v in bw.items():
                    row[k] = v
                    
                # Flatten Locality
                loc = data.get('locality', {})
                for k, v in loc.items():
                    row[f"locality_{k}"] = v
                    
                # Flatten Block Analysis
                # Create columns like block_density_4, block_density_8, etc.
                block_analysis = data.get('block_analysis', [])
                for b in block_analysis:
                    bs = b.get('block_size')
                    if bs:
                        row[f'block_density_{bs}'] = b.get('block_density')
                        row[f'nonzero_blocks_{bs}'] = b.get('nonzero_blocks')
                        row[f'total_blocks_{bs}'] = b.get('total_blocks')
                
                analysis_results.append(row)
            else:
                job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))
                print(f"Warning: No JSON in output for analysis job {job_id}", file=sys.stderr)
                
        except Exception as e:
            job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))
            print(f"Error parsing analysis job {job_id}: {e}", file=sys.stderr)

    # Export Analysis CSV
    if analysis_results:
        df_analysis = pd.DataFrame(analysis_results)
        out_file = out_dir / "results_analysis.csv"
        df_analysis.to_csv(out_file, index=False)
        print(f"Exported {len(df_analysis)} analysis rows to {out_file}")
    else:
        print("No analysis results found.")

    # --- 2. Process Operation Jobs (SpMM, etc.) ---
    op_results = []
    # Filter for operation jobs (tags containing SPMM for now, but generic enough)
    # Assuming any job that is NOT analysis is an operation job? 
    # Or stick to explicit tags. Let's stick to SPMM for now but rename variable.
    op_jobs = [j for j in all_jobs if j.tag and "SPMM" in j.tag]
    print(f"Found {len(op_jobs)} operation jobs.", file=sys.stderr)
    
    for job in op_jobs:
        try:
            # Basic Job Info
            mtx_path = safe_get_var(job, 'mtx', '')
            matrix_name = get_matrix_name(mtx_path)
            perm = safe_get_var(job, 'perm', 'None')
            tag = job.tag
            
            if 'ROW' in tag: perm_type = 'ROW'
            elif 'SYMMETRIC' in tag: perm_type = 'SYMMETRIC'
            elif 'ASYMMETRIC' in tag: perm_type = 'ASYMMETRIC'
            elif 'NO_REORDER' in tag: perm_type = 'ROW'
            else: perm_type = 'UNKNOWN'
            
            algo = tag
            block_size = safe_get_var(job, 'block_size', DEFAULT_BLOCK_SIZE, int)
            n_cols = safe_get_var(job, 'n_cols', DEFAULT_N_COLS, int)

            # Parse Timers
            timers = parse_timers(job.get_stdout())
            if not timers:
                continue
                
            job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))
            
            row = {
                'job_id': job_id,
                'tag': tag,
                'matrix': matrix_name,
                'perm': perm,
                'perm_type': perm_type,
                'algo': algo,
                'block_size': block_size if block_size > 0 else None,
                'n_cols': n_cols,
                **timers
            }
            op_results.append(row)
            
        except Exception as e:
            job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))
            print(f"Error parsing operation job {job_id}: {e}", file=sys.stderr)

    # Export Operation CSV
    if op_results:
        df_op = pd.DataFrame(op_results)
        out_file = out_dir / "results_operations.csv"
        df_op.to_csv(out_file, index=False)
        print(f"Exported {len(df_op)} operation rows to {out_file}")
    else:
        print("No operation results found.")

if __name__ == "__main__":
    main()
