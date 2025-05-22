import pandas as pd
import numpy as np
import re
import json
import unicodedata
import string
from datetime import datetime
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Any, Set, Optional

# Canonical address abbreviations
_ABBR = {
    ' avenue ': ' ave ', ' drive ': ' dr ', ' street ': ' st ',
    ' road ': ' rd ', ' crescent ': ' cres ', ' court ': ' ct ',
    ' lane ': ' ln ', ' boulevard ': ' blvd ', ' place ': ' pl ',
}

def canon_addr(addr: str) -> str:
    """Convert address to canonical form for comparison"""
    if pd.isna(addr):
        return ''
    txt = unicodedata.normalize('NFD', addr).encode('ascii','ignore').decode().lower()
    txt = txt.translate(str.maketrans('', '', string.punctuation))
    for k, v in _ABBR.items():
        txt = txt.replace(k, v)
    return re.sub(r'\s+', ' ', txt).strip()

def clean_price(s):
    """Clean price strings to numeric values"""
    return pd.to_numeric(s.astype(str).str.replace(r'[^0-9.]', '', regex=True), errors='coerce')

def split_bath(col):
    """Split bathroom count (e.g. '2:1' for 2 full, 1 half)"""
    full = col.astype(str).str.extract(r'(\d+):?')[0].astype(float)
    half = col.astype(str).str.extract(r':(\d+)')[0].astype(float)
    return full, half

def extract_city(text):
    """Extract city from address text"""
    if pd.isna(text): return np.nan
    return text.split(',')[0].strip() if ',' in text else text.strip()

def extract_province(text):
    """Extract province from address text"""
    if pd.isna(text): return np.nan
    parts = re.split(r'\s+', text.strip())
    return parts[-3] if len(parts) >= 3 else np.nan

def extract_postal(text):
    """Extract postal code from address text"""
    PC_RE = re.compile(r'[A-Za-z]\d[A-Za-z]\s?\d[A-Za-z]\d')
    m = PC_RE.search(str(text))
    return m.group(0).replace(' ', '').upper() if m else np.nan

def build_subject_row(oid, s):
    """Build a standardized row dict for a subject property"""
    return {
        'order_id'              : oid,
        'address'               : s.get('address'),
        'city'                  : s.get('municipality_district'),
        'province'              : s.get('subject_city_province_zip') or s.get('address'),
        'postal_code'           : s.get('subject_city_province_zip') or s.get('address'),
        'dom'                   : s.get('effective_date'),
        'effective_date'        : s.get('effective_date'),
        'structure_type'        : s.get('structure_type'),
        'age'                   : s.get('subject_age'),
        'gla'                   : s.get('gla'),
        'stories'               : s.get('style'),
        'bedrooms'              : s.get('num_beds'),
        'lot_size'              : f"{s.get('lot_size_sf', '')}, {s.get('site_dimensions', '')}",
        'effective_age'         : s.get('effective_age'),
        'remaining_economic_age': s.get('remaining_economic_life'),
        'basement'              : s.get('basement'),
        'basement_area'         : s.get('basement_area'),
        'heating'               : s.get('heating'),
        'cooling'               : s.get('cooling'),
        'room_count'            : s.get('room_count') or s.get('room_total'),
        'full_baths'            : s.get('num_baths'),
        'half_baths'            : s.get('num_baths'),
        'condition'             : s.get('condition'),
    }

def build_comp_row(oid, c, rank):
    """Build a standardized row dict for a comparable property"""
    return {
        'order_id'            : oid,
        'comp_rank'           : rank,
        'distance_to_subject' : c.get('distance_to_subject'),
        'structure_type'      : c.get('prop_type'),
        'stories'             : c.get('stories'),
        'address'             : c.get('address'),
        'city'                : c.get('city_province'),
        'province'            : c.get('city_province'),
        'sale_date'           : c.get('sale_date'),
        'sale_price'          : c.get('sale_price'),
        'dom'                 : c.get('dom'),
        'location_similarity' : c.get('location_similarity'),
        'lot_size'            : c.get('lot_size'),
        'age'                 : c.get('age'),
        'condition'           : c.get('condition'),
        'gla'                 : c.get('gla'),
        'room_count'          : c.get('room_count'),
        'bedrooms'            : c.get('bed_count'),
        'full_baths'          : c.get('bath_count'),
        'half_baths'          : c.get('bath_count'),
        'basement'            : c.get('basement_finish'),
        'parking'             : c.get('parking'),
        'neighborhood'        : c.get('neighborhood'),
    }

def build_property_row(oid, p):
    """Build a standardized row dict for a property in the database"""
    return {
        'order_id'    : oid,
        'id'          : p.get('id'),
        'address'     : p.get('address'),
        'bedrooms'    : p.get('bedrooms'),
        'gla'         : p.get('gla'),
        'city'        : p.get('city'),
        'province'    : p.get('province'),
        'postal_code' : p.get('postal_code'),
        'structure_type': p.get('structure_type') or p.get('property_sub_type'),
        'stories'     : p.get('style') or p.get('levels'),
        'room_count'  : p.get('room_count'),
        'full_baths'  : p.get('full_baths'),
        'half_baths'  : p.get('half_baths'),
        'lot_size'    : p.get('lot_size_sf'),
        'age'         : p.get('year_built'),
        'year_built'  : p.get('year_built'),
        'basement'    : p.get('basement'),
        'heating'     : p.get('heating'),
        'cooling'     : p.get('cooling'),
        'sale_date'   : p.get('close_date'),
        'sale_price'  : p.get('close_price'),
        'public_remarks': p.get('public_remarks'),
        'latitude'    : p.get('latitude'),
        'longitude'   : p.get('longitude'),
    }

def standardize_postal_code(value):
    if pd.isna(value):
        return None
        
    # Extract postal code pattern (A#A #A#)
    import re
    postal_pattern = r'[A-Za-z]\d[A-Za-z]\s*\d[A-Za-z]\d'
    match = re.search(postal_pattern, str(value))
    
    if match:
        # Format consistently
        postal_code = match.group(0).upper().replace(' ', '')
        # Insert space in the middle (A#A #A#)
        return postal_code[:3] + ' ' + postal_code[3:]
    
    # Alternative format K0G1J0 -> K0G 1J0
    if len(str(value)) == 6 and str(value).isalnum():
        postal_code = str(value).upper()
        return postal_code[:3] + ' ' + postal_code[3:]
        
    return value

def load_and_process_data(json_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and process the appraisal data from a JSON file
    
    Returns:
        Tuple of (subjects_df, comps_df, properties_df)
    """
    # Load the JSON data
    with open(json_path) as f:
        data = json.load(f)
    
    # Initialize lists to collect rows
    subj_rows, comp_rows, prop_rows = [], [], []
    
    # Process each appraisal
    for ap in data['appraisals']:
        oid = ap['orderID']
        subj_rows.append(build_subject_row(oid, ap['subject']))
        for i, c in enumerate(ap['comps'], 1):
            comp_rows.append(build_comp_row(oid, c, i))
        for p in ap['properties']:
            prop_rows.append(build_property_row(oid, p))
    
    # Convert to DataFrames
    df_subjects = pd.DataFrame(subj_rows)
    df_comps = pd.DataFrame(comp_rows)
    df_properties = pd.DataFrame(prop_rows)
    
    # Clean the subjects dataframe
    df_subjects['city'] = df_subjects['city'].fillna(df_subjects['address']).apply(extract_city)
    df_subjects['province'] = df_subjects['province'].apply(extract_province)
    df_subjects['postal_code'] = df_subjects['postal_code'].apply(extract_postal)
    
    # Convert date to days (DOM)
    today = pd.Timestamp('today').normalize()
    df_subjects['dom'] = (today - pd.to_datetime(df_subjects['dom'], errors='coerce')).dt.days
    
    # Convert numeric fields
    numeric_cols = ['age', 'gla', 'bedrooms', 'basement_area', 'room_count', 'effective_age', 'remaining_economic_age']
    for col in numeric_cols:
        df_subjects[col] = pd.to_numeric(df_subjects[col], errors='coerce')
    
    # Split bathroom counts
    df_subjects['full_baths'], df_subjects['half_baths'] = split_bath(df_subjects['full_baths'])
    
    # Clean the comps dataframe
    df_comps['distance_to_subject'] = pd.to_numeric(
        df_comps['distance_to_subject'].astype(str).str.replace(r'[^0-9.]','',regex=True), 
        errors='coerce'
    )
    df_comps['sale_price'] = clean_price(df_comps['sale_price'])
    df_comps['gla'] = pd.to_numeric(
        df_comps['gla'].astype(str).str.replace(r'[^0-9.]','',regex=True), 
        errors='coerce'
    )
    
    # Extract city and province
    df_comps[['city','province']] = df_comps['city'].str.extract(r'^(.*?)\\s+([A-Z]{2})')
    
    # Clean DOM
    df_comps['dom'] = pd.to_numeric(df_comps['dom'].str.replace(r'[^0-9]','',regex=True), errors='coerce')
    
    # Split bathroom counts
    df_comps['full_baths'], df_comps['half_baths'] = split_bath(df_comps['full_baths'])
    
    # Other numeric conversions
    for col in ['age', 'room_count', 'bedrooms']:
        df_comps[col] = pd.to_numeric(df_comps[col], errors='coerce')
    
    # Convert sale_date
    df_comps['sale_date'] = pd.to_datetime(df_comps['sale_date'], errors='coerce')
    
    # Clean the properties dataframe
    df_properties['sale_price'] = clean_price(df_properties['sale_price'])
    df_properties['sale_date'] = pd.to_datetime(df_properties['sale_date'], errors='coerce')
    
    # Convert numeric fields
    for c in ['gla', 'lot_size', 'room_count', 'bedrooms', 'full_baths', 'half_baths']:
        df_properties[c] = pd.to_numeric(df_properties[c], errors='coerce')
    
    # Clean stories
    df_properties['stories'] = df_properties['stories'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Calculate age from year_built
    current_year = datetime.now().year
    df_properties['year_built'] = pd.to_numeric(df_properties['year_built'], errors='coerce')
    df_properties['age'] = current_year - df_properties['year_built']
    df_properties['age'] = df_properties['age'].fillna(df_properties['age'].median())
    
    # Clean strings
    df_properties['city'] = df_properties['city'].str.strip()
    df_properties['province'] = df_properties['province'].str.strip()
    df_properties['postal_code'] = df_properties['postal_code'].apply(extract_postal)
    
    # Add canonical address for deduplication
    df_properties['canonical_address'] = df_properties['address'].apply(canon_addr)
    
    # Return the processed dataframes
    return df_subjects, df_comps, df_properties

def deduplicate_properties(df_properties: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate properties based on location and address
    
    Args:
        df_properties: DataFrame of properties
        
    Returns:
        Deduplicated properties DataFrame
    """
    # Build KD-tree for quick spatial look-ups (lat/long in radians)
    coords_rad = np.radians(df_properties[['latitude', 'longitude']].values.astype(float))
    tree = cKDTree(coords_rad)
    
    # 10 meters in radians on Earth ≈ 10 / 6371000
    eps = 10 / 6_371_000
    
    groups = {}  # idx → group_id
    current_gid = 0
    
    for idx, (addr, lat, lon) in df_properties[['canonical_address', 'latitude', 'longitude']].iterrows():
        if pd.isna(lat) or pd.isna(lon):
            continue
        if idx in groups:  # already assigned
            continue
            
        # Find candidate indices within 10 meters
        near = tree.query_ball_point(np.radians([lat, lon]), eps)
        
        # Filter to same canonical address
        dup_idxs = [i for i in near if df_properties.at[i, 'canonical_address'] == addr]
        
        if len(dup_idxs) > 1:
            for i in dup_idxs:
                groups[i] = current_gid
            current_gid += 1
    
    def pick_best(rows):
        # Pick the row with the most non-null values
        richness = rows.notna().sum(axis=1)
        best = rows.loc[richness.idxmax()]
        
        # Tie-break on sale_date (prefer most recent)
        ties = rows[richness == richness.max()]
        if len(ties) > 1 and 'sale_date' in rows.columns:
            best = ties.sort_values('sale_date', ascending=False).iloc[0]
        return best
    
    keep_indices = set()
    for gid in set(groups.values()):
        members = [i for i, g in groups.items() if g == gid]
        best_row = pick_best(df_properties.loc[members])
        keep_indices.add(best_row.name)
    
    # Build deduplicated DataFrame
    keep_df = df_properties.loc[list(keep_indices)]
    drop_df = df_properties.drop(index=list(groups.keys()))
    dedup_properties = pd.concat([keep_df, drop_df], ignore_index=True).reset_index(drop=True)
    
    return dedup_properties

def handle_missing_values(df_subjects, df_comps, df_properties):
    """
    Handle missing values in the dataframes
    
    Returns:
        Tuple of cleaned (subjects_df, comps_df, properties_df)
    """
    # Define thresholds
    row_drop_thresh = 0.10  # impute up to this rate
    col_drop_thresh = 0.50  # drop columns at or above
    
    # --- Subjects ---
    # Fill gla from basement_area first if possible
    if 'basement_area' in df_subjects:
        df_subjects.loc[
            df_subjects['gla'].isna() & df_subjects['basement_area'].notna(),
            'gla'
        ] = df_subjects['basement_area'] * 2
    
    # Check miss rate by column
    subj_miss_pct = df_subjects.isna().mean()
    
    # Drop columns with too many nulls
    drop_cols = subj_miss_pct[subj_miss_pct >= col_drop_thresh].index
    df_subjects = df_subjects.drop(columns=drop_cols)
    
    # Impute mid-range columns
    impute_cols = subj_miss_pct[(subj_miss_pct >= row_drop_thresh) & (subj_miss_pct < col_drop_thresh)].index
    for c in impute_cols:
        if pd.api.types.is_numeric_dtype(df_subjects[c]):
            df_subjects[c] = df_subjects[c].fillna(df_subjects[c].median())
        else:
            df_subjects[c] = df_subjects[c].fillna(df_subjects[c].mode().iloc[0])
    
    # Drop rows missing essential columns
    essential = ['address', 'effective_date', 'structure_type', 'gla', 'bedrooms']
    df_subjects = df_subjects.dropna(subset=essential).reset_index(drop=True)
    
    # --- Comps ---
    comp_miss_pct = df_comps.isna().mean()
    df_comps = df_comps.drop(columns=comp_miss_pct[comp_miss_pct >= col_drop_thresh].index)
    
    # Impute mid-range
    mid = comp_miss_pct[(comp_miss_pct >= row_drop_thresh) & (comp_miss_pct < col_drop_thresh)].index
    for c in mid:
        if c == 'age':
            df_comps[c] = df_comps.groupby('structure_type')[c].transform(lambda s: s.fillna(s.median()))
        elif c == 'half_baths':
            df_comps[c] = df_comps[c].fillna(0)
        elif pd.api.types.is_numeric_dtype(df_comps[c]):
            df_comps[c] = df_comps[c].fillna(df_comps[c].median())
        else:
            df_comps[c] = df_comps[c].fillna(df_comps[c].mode().iloc[0])
    
    # Drop rows missing essential columns
    must_have = ['distance_to_subject', 'sale_price', 'sale_date']
    df_comps = df_comps.dropna(subset=must_have).reset_index(drop=True)
    
    # --- Properties ---
    prop_miss_pct = df_properties.isna().mean()
    df_properties = df_properties.drop(columns=prop_miss_pct[prop_miss_pct >= col_drop_thresh].index)
    
    # Impute mid-range
    mid = prop_miss_pct[(prop_miss_pct >= row_drop_thresh) & (prop_miss_pct < col_drop_thresh)].index
    for c in mid:
        if c == 'lot_size':
            df_properties[c] = df_properties.groupby('structure_type')[c].transform(lambda s: s.fillna(s.median()))
        elif c == 'full_baths':
            bins = pd.cut(df_properties['bedrooms'], [0, 2, 4, 99], labels=['1-2', '3-4', '5+'])
            df_properties[c] = df_properties.groupby(bins)[c].transform(lambda s: s.fillna(s.median()))
        elif pd.api.types.is_numeric_dtype(df_properties[c]):
            df_properties[c] = df_properties[c].fillna(df_properties[c].median())
        else:
            df_properties[c] = df_properties[c].fillna(df_properties[c].mode().iloc[0])
    
    # Drop rows missing key columns
    must_have = ['address', 'sale_date', 'sale_price']
    df_properties = df_properties.dropna(subset=must_have).reset_index(drop=True)
    
    return df_subjects, df_comps, df_properties 