#!/usr/bin/env python3
"""
Check job status: which jobs are pending, running, completed, or failed.
Provides detailed breakdown by job type, matrix, and reordering.
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

try:
    from sbatchman import jobs_list
except ImportError:
    print("Error: sbatchman not installed. Install with: pip install sbatchman", file=sys.stderr)
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def get_matrix_name(path):
    """Extract matrix filename from path."""
    if not path:
        return "unknown"
    return Path(path).name


def safe_get_var(job, key, default=""):
    """Safely extract a variable from job.variables."""
    variables = getattr(job, 'variables', {}) or {}
    return variables.get(key, default) or default


def categorize_job(job):
    """Categorize a job by its type based on tag."""
    tag = job.tag or ""
    
    if "ANALYSIS" in tag:
        return "analysis"
    elif "SPMM" in tag or "SPMV" in tag:
        return "operation"
    else:
        return "other"


def get_job_info(job):
    """Extract relevant info from a job."""
    mtx_path = safe_get_var(job, 'mtx', '')
    matrix_name = get_matrix_name(mtx_path)
    perm = safe_get_var(job, 'perm', 'None')
    if not perm or perm == '':
        perm = 'None'
    
    tag = job.tag or ""
    
    # Determine perm_type from tag
    if 'ROW' in tag:
        perm_type = 'ROW'
    elif 'SYMMETRIC' in tag:
        perm_type = 'SYMMETRIC'
    elif 'ASYMMETRIC' in tag:
        perm_type = 'ASYMMETRIC'
    elif 'NO_REORDER' in tag:
        perm_type = 'ROW'
    else:
        perm_type = 'UNKNOWN'
    
    # Extract algo from tag (first part usually)
    algo = tag.split('_')[0] if tag else 'UNKNOWN'
    
    return {
        'matrix': matrix_name,
        'perm': perm,
        'perm_type': perm_type,
        'tag': tag,
        'algo': algo,
        'job_id': getattr(job, 'id', getattr(job, 'job_id', 'unknown')),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check job status: pending, running, completed, failed."
    )
    parser.add_argument("--tag-filter", "-t", type=str, default=None,
                        help="Filter jobs by tag (substring match)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed job information")
    parser.add_argument("--summary", "-s", action="store_true",
                        help="Show only summary")
    parser.add_argument("--show-failed", action="store_true",
                        help="Show details of failed jobs")
    parser.add_argument("--show-running", action="store_true",
                        help="Show details of running jobs")
    parser.add_argument("--show-pending", action="store_true",
                        help="Show details of pending jobs")
    parser.add_argument("--update", "-u", action="store_true",
                        help="Update job status from cluster (slower)")
    args = parser.parse_args()

    print("Fetching jobs...", file=sys.stderr)
    
    # Fetch ALL jobs in a single call (much faster than multiple calls)
    try:
        all_jobs = jobs_list(
            from_archived=True, 
            update_jobs=args.update
        )
        if args.tag_filter:
            all_jobs = [j for j in all_jobs if j.tag and args.tag_filter in j.tag]
    except Exception as e:
        print(f"Error fetching jobs: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Group jobs by status
    jobs_by_status = defaultdict(list)
    statuses = ["COMPLETED", "FAILED", "RUNNING", "PENDING", "TIMEOUT", "CANCELLED"]
    
    for job in all_jobs:
        status = getattr(job, 'status', 'UNKNOWN')
        if status in statuses:
            jobs_by_status[status].append(job)
        else:
            jobs_by_status['OTHER'].append(job)
    
    # Count totals
    total_jobs = len(all_jobs)
    
    print(f"\n{'='*60}")
    print("JOB STATUS OVERVIEW")
    print(f"{'='*60}")
    
    if args.tag_filter:
        print(f"Filter: tag contains '{args.tag_filter}'")
    
    print(f"\nTotal jobs: {total_jobs}")
    print()
    
    # Status summary
    status_colors = {
        "COMPLETED": "✓",
        "FAILED": "✗",
        "RUNNING": "►",
        "PENDING": "○",
        "TIMEOUT": "⏱",
        "CANCELLED": "⊘"
    }
    
    for status in statuses:
        count = len(jobs_by_status[status])
        if count > 0:
            icon = status_colors.get(status, "?")
            pct = 100 * count / total_jobs if total_jobs > 0 else 0
            print(f"  {icon} {status:12} {count:6} ({pct:5.1f}%)")
    
    if args.summary:
        return
    
    # ========== DETAILED BREAKDOWN BY TAG AND VARIABLES ==========
    print(f"\n{'='*60}")
    print("DETAILED BREAKDOWN BY TAG AND VARIABLES")
    print(f"{'='*60}")
    
    for status in statuses:
        jobs = jobs_by_status[status]
        if not jobs:
            continue
        
        print(f"\n{status_colors.get(status, '?')} {status} ({len(jobs)} jobs)")
        print("-" * 50)
        
        # Group by tag first
        by_tag = defaultdict(list)
        for job in jobs:
            tag = job.tag or "no_tag"
            by_tag[tag].append(job)
        
        for tag, tag_jobs in sorted(by_tag.items(), key=lambda x: -len(x[1])):
            print(f"\n  [{tag}] - {len(tag_jobs)} jobs")
            
            # Group by perm variable
            by_perm = defaultdict(list)
            for job in tag_jobs:
                perm = safe_get_var(job, 'perm', 'None')
                if not perm or perm == '':
                    perm = 'None'
                by_perm[perm].append(job)
            
            for perm, perm_jobs in sorted(by_perm.items(), key=lambda x: -len(x[1])):
                # Get unique matrices for this perm
                matrices = set()
                for job in perm_jobs:
                    mtx = safe_get_var(job, 'mtx', '')
                    matrices.add(get_matrix_name(mtx))
                
                print(f"    perm={perm}: {len(perm_jobs)} jobs ({len(matrices)} matrices)")
                
                if args.verbose:
                    for m in sorted(matrices)[:10]:
                        print(f"      - {m}")
                    if len(matrices) > 10:
                        print(f"      ... and {len(matrices) - 10} more")
    
    # ========== FAILED JOBS DETAILED ==========
    failed_jobs = jobs_by_status.get("FAILED", []) + jobs_by_status.get("TIMEOUT", [])
    
    if failed_jobs:
        print(f"\n{'='*60}")
        print(f"FAILED/TIMEOUT BREAKDOWN ({len(failed_jobs)} jobs)")
        print(f"{'='*60}")
        
        # Group by (tag, perm)
        by_tag_perm = defaultdict(list)
        for job in failed_jobs:
            tag = job.tag or "no_tag"
            perm = safe_get_var(job, 'perm', 'None')
            if not perm or perm == '':
                perm = 'None'
            by_tag_perm[(tag, perm)].append(job)
        
        # Sort by count descending
        for (tag, perm), group_jobs in sorted(by_tag_perm.items(), key=lambda x: -len(x[1])):
            matrices = set(get_matrix_name(safe_get_var(j, 'mtx', '')) for j in group_jobs)
            status_type = "TIMEOUT" if any(getattr(j, 'status', '') == 'TIMEOUT' for j in group_jobs) else "FAILED"
            print(f"\n  {status_type}: {tag}, perm={perm}")
            print(f"    Jobs: {len(group_jobs)}, Matrices: {len(matrices)}")
            
            if args.show_failed or args.verbose:
                for m in sorted(matrices)[:10]:
                    print(f"      - {m}")
                if len(matrices) > 10:
                    print(f"      ... and {len(matrices) - 10} more")
            
            # Show example error from first job in this category
            example_job = group_jobs[0]
            stderr = getattr(example_job, 'stderr', None)
            if stderr is None:
                try:
                    stderr = example_job.get_stderr()
                except:
                    stderr = None
            
            stdout = getattr(example_job, 'stdout', None)
            if stdout is None:
                try:
                    stdout = example_job.get_stdout()
                except:
                    stdout = None
            
            # Get job ID and matrix for context
            example_matrix = get_matrix_name(safe_get_var(example_job, 'mtx', ''))
            example_job_id = getattr(example_job, 'id', getattr(example_job, 'job_id', 'unknown'))
            
            print(f"\n    Example error (job {example_job_id}, matrix: {example_matrix}):")
            
            # Show last N lines of stderr or stdout
            error_text = stderr or stdout or "(no output captured)"
            if error_text and error_text != "(no output captured)":
                # Get last 10 non-empty lines
                lines = [l.strip() for l in error_text.split('\n') if l.strip()]
                last_lines = lines[-10:] if len(lines) > 10 else lines
                print("    " + "-" * 40)
                for line in last_lines:
                    # Truncate very long lines
                    if len(line) > 100:
                        line = line[:100] + "..."
                    print(f"    | {line}")
                print("    " + "-" * 40)
            else:
                print(f"    {error_text}")
    
    # ========== COMPLETED BREAKDOWN ==========
    completed_jobs = jobs_by_status.get("COMPLETED", [])
    
    if completed_jobs and args.verbose:
        print(f"\n{'='*60}")
        print(f"COMPLETED BREAKDOWN ({len(completed_jobs)} jobs)")
        print(f"{'='*60}")
        
        # Group by (tag, perm)
        by_tag_perm = defaultdict(list)
        for job in completed_jobs:
            tag = job.tag or "no_tag"
            perm = safe_get_var(job, 'perm', 'None')
            if not perm or perm == '':
                perm = 'None'
            by_tag_perm[(tag, perm)].append(job)
        
        for (tag, perm), group_jobs in sorted(by_tag_perm.items(), key=lambda x: -len(x[1])):
            matrices = set(get_matrix_name(safe_get_var(j, 'mtx', '')) for j in group_jobs)
            print(f"  COMPLETED: {tag}, perm={perm} - {len(group_jobs)} jobs ({len(matrices)} matrices)")
    
    # ========== RUNNING/PENDING BREAKDOWN ==========
    running_jobs = jobs_by_status.get("RUNNING", [])
    pending_jobs = jobs_by_status.get("PENDING", [])
    
    if running_jobs and (args.show_running or args.verbose):
        print(f"\n{'='*60}")
        print(f"RUNNING BREAKDOWN ({len(running_jobs)} jobs)")
        print(f"{'='*60}")
        
        by_tag_perm = defaultdict(list)
        for job in running_jobs:
            tag = job.tag or "no_tag"
            perm = safe_get_var(job, 'perm', 'None') or 'None'
            by_tag_perm[(tag, perm)].append(job)
        
        for (tag, perm), group_jobs in sorted(by_tag_perm.items(), key=lambda x: -len(x[1])):
            matrices = set(get_matrix_name(safe_get_var(j, 'mtx', '')) for j in group_jobs)
            print(f"  RUNNING: {tag}, perm={perm} - {len(group_jobs)} jobs ({len(matrices)} matrices)")
    
    if pending_jobs and (args.show_pending or args.verbose):
        print(f"\n{'='*60}")
        print(f"PENDING BREAKDOWN ({len(pending_jobs)} jobs)")
        print(f"{'='*60}")
        
        by_tag_perm = defaultdict(list)
        for job in pending_jobs:
            tag = job.tag or "no_tag"
            perm = safe_get_var(job, 'perm', 'None') or 'None'
            by_tag_perm[(tag, perm)].append(job)
        
        for (tag, perm), group_jobs in sorted(by_tag_perm.items(), key=lambda x: -len(x[1])):
            matrices = set(get_matrix_name(safe_get_var(j, 'mtx', '')) for j in group_jobs)
            print(f"  PENDING: {tag}, perm={perm} - {len(group_jobs)} jobs ({len(matrices)} matrices)")
    
    # ========== CROSS-CHECK ==========
    print(f"\n{'='*60}")
    print("CROSS-CHECK: OPERATIONS vs ANALYSIS")
    print(f"{'='*60}")
    
    completed_jobs = jobs_by_status.get("COMPLETED", [])
    
    # Separate by category
    completed_ops = [j for j in completed_jobs if categorize_job(j) == "operation"]
    completed_analysis = [j for j in completed_jobs if categorize_job(j) == "analysis"]
    
    print(f"\nCompleted operations: {len(completed_ops)}")
    print(f"Completed analysis: {len(completed_analysis)}")
    
    # Get unique (matrix, perm, perm_type) from operations
    op_keys = set()
    for job in completed_ops:
        info = get_job_info(job)
        op_keys.add((info['matrix'], info['perm'], info['perm_type']))
    
    # Get unique (matrix, perm, perm_type) from analysis  
    analysis_keys = set()
    for job in completed_analysis:
        info = get_job_info(job)
        analysis_keys.add((info['matrix'], info['perm'], info['perm_type']))
    
    # Check for operations without analysis
    ops_without_analysis = op_keys - analysis_keys
    
    if ops_without_analysis:
        print(f"\n[WARNING] {len(ops_without_analysis)} operation configs lack completed analysis:")
        
        # Group by (perm, perm_type)
        by_reorder = defaultdict(list)
        for matrix, perm, perm_type in ops_without_analysis:
            by_reorder[(perm, perm_type)].append(matrix)
        
        for (perm, perm_type), matrices in sorted(by_reorder.items()):
            print(f"\n  perm={perm}, type={perm_type}: {len(matrices)} matrices")
            if args.verbose:
                for m in sorted(matrices):
                    print(f"    - {m}")
            elif len(matrices) <= 5:
                print(f"    {', '.join(sorted(matrices))}")
            else:
                print(f"    {', '.join(sorted(matrices)[:5])} ... and {len(matrices)-5} more")
    else:
        print("\n[OK] All completed operations have corresponding analysis.")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    completed = len(jobs_by_status.get("COMPLETED", []))
    failed = len(jobs_by_status.get("FAILED", [])) + len(jobs_by_status.get("TIMEOUT", []))
    running = len(jobs_by_status.get("RUNNING", []))
    pending = len(jobs_by_status.get("PENDING", []))
    
    if total_jobs > 0:
        print(f"\nCompletion rate: {100*completed/total_jobs:.1f}%")
        
        if failed > 0:
            print(f"Failure rate: {100*failed/total_jobs:.1f}%")
        
        if running > 0 or pending > 0:
            print(f"In progress: {running} running, {pending} pending")
    
    if failed > 0:
        print(f"\n[ACTION] {failed} jobs need attention (failed/timeout)")
        print("  Run with --show-failed for details")


if __name__ == "__main__":
    main()
