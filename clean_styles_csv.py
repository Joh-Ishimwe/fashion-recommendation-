# clean_styles_csv.py
import pandas as pd
import csv
import numpy as np

# Read the CSV with proper quoting and skip bad lines
df = pd.read_csv('data/styles.csv', quoting=csv.QUOTE_ALL, on_bad_lines='skip')

# Ensure only the required columns are kept
required_columns = ['gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year']
df = df[required_columns]

# Handle missing values in 'season' and convert to string
df['season'] = df['season'].fillna('Unknown').astype(str)

# Define a more nuanced mapping for 'usage' based on multiple features
def assign_usage(row):
    # Base mapping on articleType
    if row['articleType'] in ['Tshirts', 'Jeans', 'Dresses', 'Casual Shoes']:
        base_usage = 'Casual'
    elif row['articleType'] in ['Shirts', 'Trousers', 'Formal Shoes']:
        base_usage = 'Formal'
    elif row['articleType'] in ['Watches', 'Sunglasses', 'Belts']:
        base_usage = 'Fashionable'
    else:
        base_usage = 'Casual'

    # Adjust based on season (example: summer items are more likely to be Casual)
    season_lower = row['season'].lower()
    if season_lower == 'summer' and np.random.rand() > 0.2:  # 80% chance
        return 'Casual'
    elif season_lower == 'winter' and np.random.rand() > 0.5:  # 50% chance
        return 'Formal'
    
    return base_usage

# Assign 'usage' with some randomness to avoid deterministic mapping
df['usage'] = df.apply(assign_usage, axis=1)

# Debug: Print the class distribution of 'usage'
print("Class distribution in 'usage':")
print(df['usage'].value_counts())

# Save the cleaned CSV
df.to_csv('data/styles_cleaned.csv', index=False, quoting=csv.QUOTE_ALL)
print("Cleaned CSV saved as data/styles_cleaned.csv")