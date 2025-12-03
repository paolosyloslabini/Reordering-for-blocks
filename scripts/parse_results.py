import pandas as pd
import json
import re
import os
import sys
from pathlib import Path
from sbatchman import jobs_list

def get_matrix_name(path):
    """Extract matrix filename from path."""
    return Path(path).name

def parse_timers(stdout):
    """Extract all timers from stdout."""
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
    print("Fetching jobs...", file=sys.stderr)
    
    # Fetch ALL completed jobs to avoid potential wildcard issues in sbatchman query
    try:
        all_jobs = jobs_list(from_archived=True, status=["COMPLETED"])
    except Exception as e:
        print(f"Error fetching jobs: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Total completed jobs found: {len(all_jobs)}", file=sys.stderr)
    
    if len(all_jobs) == 0:
        print("No completed jobs found. Have you run the experiments yet?", file=sys.stderr)
        sys.exit(0)

    # Debug: print unique tags found
    unique_tags = set(j.tag for j in all_jobs)
    print(f"Found tags: {sorted(list(unique_tags))}", file=sys.stderr)
    
    # 1. Collect Analysis Results
    # Key: (matrix_name, perm_name, perm_type) -> data dict
    analysis_cache = {}
    
    # Filter for analysis jobs (tags starting with ANALYSIS_)
    analysis_jobs = [j for j in all_jobs if j.tag and j.tag.startswith("ANALYSIS_")]
    print(f"Found {len(analysis_jobs)} analysis jobs.", file=sys.stderr)
    
    for job in analysis_jobs:
        try:
            # Parse variables
            mtx_path = job.variables.get('mtx', '')
            matrix_name = get_matrix_name(mtx_path)
            perm = job.variables.get('perm', 'None')
            
            # Determine perm_type from tag or command
            # The tag is usually ANALYSIS_ROW, ANALYSIS_SYMMETRIC, etc.
            tag = job.tag
            if 'ROW' in tag:
                perm_type = 'ROW'
            elif 'SYMMETRIC' in tag:
                perm_type = 'SYMMETRIC'
            elif 'ASYMMETRIC' in tag:
                perm_type = 'ASYMMETRIC'
            elif 'NO_REORDER' in tag:
                perm_type = 'ROW' # Default to ROW for no reorder
            else:
                perm_type = 'UNKNOWN'
            
            # Parse JSON output
            stdout = job.get_stdout()
            # Find the JSON part - it might be surrounded by other logs if not using --pretty
            # But our script prints JSON to stdout.
            # If there are warnings (like cupy warning), they might be in stdout if not redirected?
            # Usually warnings go to stderr.
            # Let's try to find the first { and last }
            start = stdout.find('{')
            end = stdout.rfind('}')
            
            if start != -1 and end != -1:
                json_str = stdout[start:end+1]
                data = json.loads(json_str)
                
                # Store in cache
                key = (matrix_name, perm, perm_type)
                analysis_cache[key] = data
            else:
                # Try to get job ID safely for logging
                job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))
                print(f"Warning: Could not find JSON in output for job {job_id} ({tag})", file=sys.stderr)
                
        except Exception as e:
            job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))
            print(f"Error parsing analysis job {job_id}: {e}", file=sys.stderr)

    print(f"Cached analysis results for {len(analysis_cache)} configurations.", file=sys.stderr)

    # 2. Collect Multiplication Results
    results = []
    
    # Filter for SpMM jobs (tags containing SPMM)
    spmm_jobs = [j for j in all_jobs if j.tag and "SPMM" in j.tag]
    print(f"Found {len(spmm_jobs)} SpMM jobs.", file=sys.stderr)
    
    for job in spmm_jobs:
        try:
            # Basic Job Info
            mtx_path = job.variables.get('mtx', '')
            matrix_name = get_matrix_name(mtx_path)
            perm = job.variables.get('perm', 'None')
            
            # Determine perm_type and algo from tag
            tag = job.tag
            
            if 'ROW' in tag:
                perm_type = 'ROW'
            elif 'SYMMETRIC' in tag:
                perm_type = 'SYMMETRIC'
            elif 'ASYMMETRIC' in tag:
                perm_type = 'ASYMMETRIC'
            elif 'NO_REORDER' in tag:
                perm_type = 'ROW' # Default to ROW for no reorder
            else:
                perm_type = 'UNKNOWN'
                
            # Algorithm
            # Use the tag directly as the algorithm identifier
            # This makes it future-proof for new algorithms
            algo = tag
            
            # Block Size (only relevant for BSR/SMAT)
            # Try to get from variables, default to 0
            try:
                block_size = int(job.variables.get('block_size', 0))
            except (ValueError, TypeError):
                block_size = 0
            
            # N_COLS (Dense matrix columns)
            try:
                n_cols = int(job.variables.get('n_cols', 32))
            except (ValueError, TypeError):
                n_cols = 32

            # Parse Timers
            timers = parse_timers(job.get_stdout())
            if not timers:
                # Skip failed jobs or those without output
                continue
                
            # Build Row
            # Note: sbatchman Job object might use job_id instead of id
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
            
            # 3. Merge with Analysis Data
            analysis_key = (matrix_name, perm, perm_type)
            if analysis_key in analysis_cache:
                adata = analysis_cache[analysis_key]
                
                # General Metrics
                row['rows'] = adata.get('rows')
                row['cols'] = adata.get('cols')
                row['nnz'] = adata.get('nnz')
                row['density'] = adata.get('density')
                
                # Bandwidth
                bw = adata.get('bandwidth', {})
                row['bandwidth_max'] = bw.get('bandwidth_max')
                row['bandwidth_avg'] = bw.get('bandwidth_avg')
                
                # Locality
                loc = adata.get('locality', {})
                row['locality_profile'] = loc.get('profile')
                row['locality_avg_row_spread'] = loc.get('avg_row_spread')
                
                # Block Analysis (Specific to this job's block_size)
                if block_size > 0:
                    block_stats = next((b for b in adata.get('block_analysis', []) if b['block_size'] == block_size), None)
                    if block_stats:
                        row['block_density'] = block_stats.get('block_density')
                        row['nonzero_blocks'] = block_stats.get('nonzero_blocks')
                        row['total_blocks'] = block_stats.get('total_blocks')
            
            results.append(row)
            
        except Exception as e:
            job_id = getattr(job, 'id', getattr(job, 'job_id', 'unknown'))
            print(f"Error parsing SpMM job {job_id}: {e}", file=sys.stderr)

    # 4. Export to CSV
    if results:
        df = pd.DataFrame(results)
        output_file = "results_combined.csv"
        df.to_csv(output_file, index=False)
        print(f"Successfully exported {len(df)} rows to {output_file}")
        
        # Print a preview
        print("\nPreview:")
        print(df[['matrix', 'algo', 'perm', 'time_operation_ms', 'block_density']].head())
    else:
        print("No results found to export.")

if __name__ == "__main__":
    main()

