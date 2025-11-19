#collects all jobs_csv into a csv file

def main():
  parser = argparse.ArgumentParser(description="Generate comparative scaling plots from HPL benchmark CSV files.")
  parser.add_argument("csv_files", nargs="+", help="Paths to CSV files with HPL results")
  parser.add_argument("--out", default="results", help="Directory for output plot files")
  args = parser.parse_args()

  # Read all CSVs and concatenate
  dfs = [pd.read_csv(f) for f in args.csv_files]
  df = pd.concat(dfs, ignore_index=True)