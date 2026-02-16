#!/usr/bin/env python3
"""
Run data ingestion pipeline with coordinate averaging.
This script:
1. Fetches teams, players, and games from UFA API
2. Cleans and processes game events
3. Inserts missing turnovers
4. Averages turnover coordinates (NEW!)
5. Inserts everything into PostgreSQL
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
print("Starting Data Ingestion Pipeline")
print("="*80)

# Configuration
CONFIG = {
    'year': '2024',
    'date_range': '2024-04:2024-05',
    'game_status': 'Final',
    'skip_existing': False,  # Reprocess all games
}

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# This will execute the ingestion
print("\nThis will ingest data with the NEW coordinate averaging!")
print("Press Ctrl+C within 3 seconds to cancel...\n")

import time
time.sleep(3)

print("Starting ingestion...")
