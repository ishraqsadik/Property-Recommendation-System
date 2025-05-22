import os
import json
import pandas as pd
from collections import Counter
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def analyze_json_structure(data):
    """
    Analyze the structure of the JSON data
    
    Args:
        data: Loaded JSON data
    """
    print("\n=== JSON STRUCTURE ANALYSIS ===")
    
    # Check if it's a list or dictionary
    data_type = type(data)
    print(f"Data type: {data_type}")
    
    if isinstance(data, list):
        print(f"Number of records: {len(data)}")
        
        # Analyze the first few records
        for i, record in enumerate(data[:3]):
            print(f"\nRecord {i+1} structure:")
            analyze_record_structure(record)
    
    elif isinstance(data, dict):
        print(f"Keys at root level: {list(data.keys())}")
        
        # Analyze each key at the root level
        for key in data.keys():
            value = data[key]
            print(f"\nKey: {key}")
            print(f"Type: {type(value)}")
            
            if isinstance(value, list):
                print(f"List length: {len(value)}")
                if len(value) > 0:
                    print("First item structure:")
                    analyze_record_structure(value[0])
            
            elif isinstance(value, dict):
                print(f"Dict keys: {list(value.keys())}")

def analyze_record_structure(record, prefix=""):
    """
    Analyze the structure of a single record
    
    Args:
        record: Single JSON record
        prefix: Prefix for nested keys
    """
    if not isinstance(record, dict):
        print(f"{prefix}Value type: {type(record)}")
        return
    
    for key, value in record.items():
        full_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            print(f"{full_key}: dict with keys {list(value.keys())}")
            analyze_record_structure(value, full_key)
        
        elif isinstance(value, list):
            print(f"{full_key}: list with {len(value)} items")
            if len(value) > 0 and isinstance(value[0], dict):
                print(f"  First item has keys: {list(value[0].keys())}")
        
        else:
            print(f"{full_key}: {type(value)} = {value}")

def analyze_value_distributions(data):
    """
    Analyze the distributions of values in the data
    
    Args:
        data: Loaded JSON data
    """
    print("\n=== VALUE DISTRIBUTIONS ===")
    
    # Extract all orders
    orders = []
    if isinstance(data, list):
        orders = data
    elif isinstance(data, dict) and 'orders' in data:
        orders = data['orders']
    
    if not orders:
        print("No orders found in data")
        return
    
    # Collect all property types
    property_types = []
    subject_property_types = []
    
    # Collect all field names and their value types
    field_names = Counter()
    field_value_types = {}
    
    for order in orders[:100]:  # Limit to first 100 for speed
        # Subject property
        if 'subject' in order:
            subject = order['subject']
            if isinstance(subject, dict):
                for key, value in subject.items():
                    field_names[f"subject.{key}"] += 1
                    field_value_types.setdefault(f"subject.{key}", set()).add(type(value).__name__)
                
                if 'property_type' in subject:
                    subject_property_types.append(subject['property_type'])
        
        # Comparables
        if 'comparables' in order:
            comps = order['comparables']
            if isinstance(comps, list):
                for i, comp in enumerate(comps[:5]):  # Limit to first 5 comps
                    if isinstance(comp, dict):
                        for key, value in comp.items():
                            field_names[f"comparables.{key}"] += 1
                            field_value_types.setdefault(f"comparables.{key}", set()).add(type(value).__name__)
                        
                        if 'property_type' in comp:
                            property_types.append(comp['property_type'])
    
    # Print property type distributions
    print("\nSubject Property Types:")
    subject_type_counts = Counter(subject_property_types)
    for prop_type, count in subject_type_counts.most_common():
        print(f"  {prop_type}: {count}")
    
    print("\nComparable Property Types:")
    comp_type_counts = Counter(property_types)
    for prop_type, count in comp_type_counts.most_common():
        print(f"  {prop_type}: {count}")
    
    # Print field distributions
    print("\nField Names and Counts:")
    for field, count in field_names.most_common():
        types = field_value_types.get(field, set())
        print(f"  {field}: {count} occurrences, types: {types}")

def main():
    """
    Load and analyze the raw JSON data
    """
    # Path to the JSON data file
    json_path = "appraisals_dataset.json"
    
    if not os.path.exists(json_path):
        logger.error(f"Data file not found at {json_path}")
        return
    
    logger.info(f"Loading raw data from {json_path}")
    
    try:
        # Load raw JSON data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Analyze the data structure
        analyze_json_structure(data)
        
        # Analyze value distributions
        analyze_value_distributions(data)
        
        # Save analysis to file
        with open("raw_data_analysis.txt", "w") as f:
            # Redirect stdout to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            analyze_json_structure(data)
            analyze_value_distributions(data)
            
            # Restore stdout
            sys.stdout = original_stdout
        
        logger.info("Analysis complete. Results saved to raw_data_analysis.txt")
        
    except Exception as e:
        logger.error(f"Error analyzing raw data: {e}")

if __name__ == "__main__":
    main() 