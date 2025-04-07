# clean_csv.py
import csv
import pandas as pd

# Read the CSV with error handling
try:
    df = pd.read_csv('data/styles.csv', on_bad_lines='skip', quoting=csv.QUOTE_ALL)
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit(1)

# Expected columns
expected_columns = ['id', 'gender', 'masterCategory', 'subCategory', 'articleType', 'baseColour', 'season', 'year', 'usage', 'productDisplayName']

# Check if all expected columns are present
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
    # If 'usage' is missing, add a dummy value
    if 'usage' in missing_columns:
        df['usage'] = 'Casual'  # Add a dummy value for usage

# Save the cleaned CSV
df.to_csv('data/styles_cleaned.csv', index=False, quoting=csv.QUOTE_ALL)
print("Cleaned CSV saved as 'data/styles_cleaned.csv'")