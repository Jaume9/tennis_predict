import os
import pandas as pd
import glob
from pathlib import Path

# Base directory for data
DATA_DIR = Path("data")

def load_atp_data(start_year=2010, end_year=2023):
    """Load ATP match data for a range of years"""
    print(f"Loading ATP data from {start_year} to {end_year}...")
    all_files = []
    
    # Find files matching pattern atp_matches_YYYY.csv
    for year in range(start_year, end_year + 1):
        pattern = DATA_DIR / f"atp_matches_{year}.csv"
        files = glob.glob(str(pattern))
        all_files.extend(files)
    
    if not all_files:
        raise ValueError(f"No files found for years {start_year}-{end_year}")
    
    # Load and combine files
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            print(f"Loaded: {os.path.basename(file)}, {df.shape[0]} records")
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined ATP data: {combined_df.shape[0]} matches")
    
    return combined_df