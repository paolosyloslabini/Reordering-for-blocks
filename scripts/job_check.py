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
    
    # Categorize jobs by type
    print(f"\n{'='*60}")
    print("BREAKDOWN BY JOB TYPE")
    print(f"{'='*60}")
    
    for status in statuses:
        jobs = jobs_by_status[status]
        if not jobs:
            continue
        
        by_category = defaultdict(list)
        for job in jobs:
            category = categorize_job(job)
            by_category[category].append(job)
        
        print(f"\n{status}:")
        for category, cat_jobs in sorted(by_category.items()):
            print(f"  {category}: {len(cat_jobs)}")
    
    # Failed jobs details
    failed_jobs = jobs_by_status.get("FAILED", []) + jobs_by_status.get("TIMEOUT", [])
    
    if failed_jobs and (args.show_failed or args.verbose):
        print(f"\n{'='*60}")
        print(f"FAILED/TIMEOUT JOBS ({len(failed_jobs)})")
        print(f"{'='*60}")
        
        # Group by tag pattern
        by_tag = defaultdict(list)
        for job in failed_jobs:
            tag = job.tag or "no_tag"
            by_tag[tag].append(job)
        
        for tag, tag_jobs in sorted(by_tag.items()):
            print(f"\n  Tag: {tag}")
            print(f"  Count: {len(tag_jobs)}")
            
            # Group by matrix
            by_matrix = defaultdict(list)
            for job in tag_jobs:
                info = get_job_info(job)
                by_matrix[info['matrix']].append(job)
            
            if args.verbose:
                print(f"  Matrices:")
                for matrix, matrix_jobs in sorted(by_matrix.items()):
                    print(f"    - {matrix} ({len(matrix_jobs)} jobs)")
            else:
                print(f"  Affects {len(by_matrix)} matrices")
                matrices = sorted(by_matrix.keys())
                if len(matrices) <= 5:
                    print(f"    {', '.join(matrices)}")
                else:
                    print(f"    {', '.join(matrices[:5])} ... and {len(matrices)-5} more")
    
    # Running jobs details
    running_jobs = jobs_by_status.get("RUNNING", [])
    
    if running_jobs and (args.show_running or args.verbose):
        print(f"\n{'='*60}")
        print(f"RUNNING JOBS ({len(running_jobs)})")
        print(f"{'='*60}")
        
        by_tag = defaultdict(list)
        for job in running_jobs:
            tag = job.tag or "no_tag"
            by_tag[tag].append(job)
        
        for tag, tag_jobs in sorted(by_tag.items()):
            print(f"\n  Tag: {tag} ({len(tag_jobs)} jobs)")
            if args.verbose:
                for job in tag_jobs[:10]:
                    info = get_job_info(job)
                    print(f"    - {info['matrix']} (perm={info['perm']})")
                if len(tag_jobs) > 10:
                    print(f"    ... and {len(tag_jobs) - 10} more")
    
    # Pending jobs details
    pending_jobs = jobs_by_status.get("PENDING", [])
    
    if pending_jobs and (args.show_pending or args.verbose):
        print(f"\n{'='*60}")
        print(f"PENDING JOBS ({len(pending_jobs)})")
        print(f"{'='*60}")
        
        by_tag = defaultdict(list)
        for job in pending_jobs:
            tag = job.tag or "no_tag"
            by_tag[tag].append(job)
        
        for tag, tag_jobs in sorted(by_tag.items()):
            print(f"\n  Tag: {tag} ({len(tag_jobs)} jobs)")
    
    # Missing analysis for completed operations
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
