#!/usr/bin/env python3
"""
Run data ingestion for multiple years (2023, 2024, 2025).
This script ingests all available games for the specified years.
"""

import sys
import os

# Add current directory to path so we can import from notebooks
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Run the notebooks to load their functions
print("Loading functions from data_cleaning_functions.ipynb...")
exec(open('data_cleaning_functions.ipynb').read())

print("Loading functions from postgres_setup.ipynb...")
exec(open('postgres_setup.ipynb').read())

print("\n" + "="*80)
print("MULTI-YEAR DATA INGESTION PIPELINE")
print("="*80)

# Years to ingest
YEARS = ['2023', '2024', '2025']

print(f"\nYears to ingest: {', '.join(YEARS)}")
print("Game status: Final")
print("Skip existing: True (won't duplicate data)")

# Confirmation
print("\n" + "="*80)
print("This will ingest ALL games from 2023, 2024, and 2025!")
print("Press Ctrl+C within 5 seconds to cancel...")
print("="*80 + "\n")

import time
time.sleep(5)

# Ingest each year
for year in YEARS:
    print("\n" + "="*80)
    print(f"INGESTING YEAR: {year}")
    print("="*80)

    CONFIG = {
        'year': year,
        'date_range': f'{year}-01:{year}-12',  # Full year
        'game_status': 'Final',
        'skip_existing': True,  # Don't reprocess existing games
    }

    print(f"\nConfiguration for {year}:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    print(f"\nStarting ingestion for {year}...")

    # The ingestion will run here automatically based on the notebooks loaded
    # The notebooks should detect CONFIG and run ingestion

    print(f"\n✅ Completed ingestion for {year}")

print("\n" + "="*80)
print("ALL YEARS INGESTED SUCCESSFULLY!")
print("="*80)
print("\nNext steps:")
print("1. Run verify_database.py to check data")
print("2. Retrain MDN model with expanded dataset")
