import os
import pandas as pd
import numpy as np
import json
from collections import Counter
import logging

from app.ml.data_processing import load_and_process_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def analyze_dataframe(df, name):
    """
    Analyze a dataframe by printing information about its columns and unique values
    
    Args:
        df: DataFrame to analyze
        name: Name of the dataframe for printing
    """
    print(f"\n{'=' * 80}")
    print(f"ANALYZING {name.upper()} DATAFRAME")
    print(f"{'=' * 80}")
    
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    print("\nColumn Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    print("\nUnique Values for Each Column:")
    for col in df.columns:
        try:
            if df[col].dtype == 'object' or df[col].nunique() < 50:
                unique_values = df[col].dropna().unique()
                if len(unique_values) < 30:  # Only print if not too many values
                    print(f"\n{col}: {unique_values}")
                else:
                    # Get value counts for most common values
                    value_counts = df[col].value_counts().head(10)
                    print(f"\n{col}: {len(unique_values)} unique values")
                    print(f"Top 10 values: {value_counts.to_dict()}")
            else:
                # For numeric columns with many values, print statistics
                print(f"\n{col}: {df[col].nunique()} unique values")
                print(f"Min: {df[col].min()}, Max: {df[col].max()}, Mean: {df[col].mean():.2f}, Median: {df[col].median():.2f}")
        except Exception as e:
            print(f"Error analyzing column {col}: {e}")

def main():
    """
    Load and analyze the dataset
    """
    # Path to the JSON data file
    json_path = "appraisals_dataset.json"
    
    if not os.path.exists(json_path):
        logger.error(f"Data file not found at {json_path}")
        return
    
    logger.info(f"Loading data from {json_path}")
    
    try:
        # Load and process data
        subjects_df, comps_df, properties_df = load_and_process_data(json_path)
        
        # Analyze each dataframe
        analyze_dataframe(subjects_df, "Subjects")
        analyze_dataframe(comps_df, "Comps")
        analyze_dataframe(properties_df, "Properties")
        
        # Save analysis to file
        with open("data_analysis_results.txt", "w") as f:
            # Redirect stdout to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            analyze_dataframe(subjects_df, "Subjects")
            analyze_dataframe(comps_df, "Comps")
            analyze_dataframe(properties_df, "Properties")
            
            # Restore stdout
            sys.stdout = original_stdout
        
        logger.info("Analysis complete. Results saved to data_analysis_results.txt")
        
    except Exception as e:
        logger.error(f"Error analyzing data: {e}")

if __name__ == "__main__":
    main() 