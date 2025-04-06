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

# Define usage distribution as per your expectation
usage_distribution = {
    'Casual': 0.3,
    'Sports': 0.2,
    'Formal': 0.15,
    'Ethnic': 0.1,
    'Party': 0.1,
    'Travel': 0.05,
    'Smart Casual': 0.05,
    'Home': 0.05
}

# Define a nuanced mapping for 'usage' based on multiple features
def assign_usage(row):
    article = row['articleType'].lower()
    master = row['masterCategory'].lower()
    sub = row['subCategory'].lower()
    season = row['season'].lower()

    # Probabilistic assignment based on features
    rand = np.random.rand()
    if 'sports' in article or 'sports' in sub or 'sportswear' in master:
        return 'Sports' if rand < 0.8 else 'Casual'
    elif 'ethnic' in article or 'kurtas' in article or 'sarees' in article:
        return 'Ethnic' if rand < 0.7 else 'Party'
    elif 'formal' in article or 'shirts' in article or 'trousers' in article or 'formal shoes' in article:
        return 'Formal' if rand < 0.6 else 'Smart Casual'
    elif 'party' in article or 'dresses' in article or 'heels' in article:
        return 'Party' if rand < 0.6 else 'Smart Casual'  # Changed from Fashionable
    elif 'loungewear' in article or 'home' in sub or 'sleepwear' in master:
        return 'Home' if rand < 0.7 else 'Casual'
    elif 'travel' in article or 'backpacks' in article or 'luggage' in sub:
        return 'Travel' if rand < 0.6 else 'Casual'
    elif 'casual' in article or 'tshirts' in article or 'jeans' in article or 'casual shoes' in article:
        return 'Casual' if rand < 0.7 else 'Smart Casual'
    elif 'watch' in article or 'sunglasses' in article or 'belt' in article:
        return 'Smart Casual' if rand < 0.6 else 'Casual'  # Reassigned Fashionable

    # Fallback with weighted random choice
    usages = list(usage_distribution.keys())
    weights = list(usage_distribution.values())
    return np.random.choice(usages, p=weights)

# Assign 'usage' to the DataFrame
df['usage'] = df.apply(assign_usage, axis=1)

# Debug: Print the class distribution of 'usage'
print("Class distribution in 'usage':")
print(df['usage'].value_counts())

# Save the cleaned CSV
df.to_csv('data/styles_cleaned.csv', index=False, quoting=csv.QUOTE_ALL)
print("Cleaned CSV saved as data/styles_cleaned.csv")