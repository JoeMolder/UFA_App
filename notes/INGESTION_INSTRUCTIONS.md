# Multi-Year Data Ingestion Instructions

## Simplest Approach: Run Notebook 3 Times

Open `data_ingest.ipynb` in Jupyter and run for each year:

### Year 1: 2023
1. Find the configuration cell (near the top)
2. Update to:
   ```python
   year = '2023'
   date_range = '2023-01:2023-12'
   game_status = 'Final'
   ```
3. Run all cells
4. Wait for completion

### Year 2: 2024
1. Update configuration:
   ```python
   year = '2024'
   date_range = '2024-01:2024-12'
   game_status = 'Final'
   ```
2. Run all cells
3. Wait for completion

### Year 3: 2025
1. Update configuration:
   ```python
   year = '2025'
   date_range = '2025-01:2025-12'
   game_status = 'Final'
   ```
2. Run all cells
3. Wait for completion

## Verify After Completion

```bash
python3 verify_database.py
```

Should show significantly more:
- Total games (from all 3 years)
- Total events (100k-300k+)
- Synthetic events

## Then: Retrain MDN

Once ingestion completes, go back to `mdn_with_embeddings.ipynb` and:
1. Re-run data loading cells (will load ALL data)
2. Retrain model with expanded dataset
3. Compare results!
