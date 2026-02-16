#!/bin/bash
# Run data ingestion with corrected orientation-aware coordinate averaging

echo "=================================================================="
echo "Running Data Ingestion with Orientation-Corrected Averaging"
echo "=================================================================="
echo ""

cd /Users/joemolder/Documents/UFAnalysis

# Use Python to execute the notebook cells programmatically
python3 << 'PYTHON_SCRIPT'
import sys
import subprocess

# Convert notebook to Python script and run it
print("Executing data ingestion pipeline...")
result = subprocess.run([
    'jupyter', 'nbconvert',
    '--to', 'script',
    '--execute',
    'data_ingest.ipynb',
    '--output', '/tmp/data_ingest_run.py'
], capture_output=True, text=True)

if result.returncode != 0:
    print("Error executing notebook:")
    print(result.stderr)
    sys.exit(1)

print("\n✅ Data ingestion complete!")

PYTHON_SCRIPT

echo ""
echo "=================================================================="
echo "Verifying results..."
echo "=================================================================="
python3 verify_database.py

echo ""
echo "=================================================================="
echo "Checking for midfield clustering..."
echo "=================================================================="
python3 check_averaging_issue.py
