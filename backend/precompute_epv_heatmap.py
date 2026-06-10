"""
Precompute EPV heatmaps for all (throw_idx, team, quarter) combinations.

Schema: epv_heatmap_cache(throw_idx, team, quarter, grid BYTEA)
grid = float16 array shape (60, 25) = 3,000 bytes per row
Combinations: 10 throw_idx × ~21 teams × 5 quarters = ~1,050 rows (~3MB)

team='' means league average (no team filter)
quarter=0 means all quarters averaged

Run: python precompute_epv_heatmap.py
Set DATABASE_URL env var to target Railway, or uses local DB by default.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import joblib
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from dotenv import load_dotenv
import urllib.parse
import time

load_dotenv()

# --- DB connection ---
_database_url = os.getenv("DATABASE_URL")
if _database_url:
    _u = urllib.parse.urlparse(_database_url)
    DB_CONFIG = {
        'dbname': _u.path.lstrip('/'),
        'user': _u.username,
        'password': _u.password,
        'host': _u.hostname,
        'port': _u.port or 5432,
    }
else:
    DB_CONFIG = {
        'dbname': os.getenv('DB_NAME', 'ufa_analytics'),
        'user': os.getenv('DB_USER', 'joemolder'),
        'password': os.getenv('DB_PASSWORD', ''),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
    }

# --- Model architecture ---
class EPVNet(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

# --- Load model ---
EPV_PATH = Path(__file__).resolve().parent / "models" / "epv_nn.pkl"
print(f"Loading EPV model from {EPV_PATH}...")
epv_save = joblib.load(EPV_PATH)
scaler = epv_save["scaler"]
team_encoder = epv_save["team_encoder"]
model = EPVNet(input_dim=epv_save.get("input_dim", 6))
model.load_state_dict(epv_save["model_state_dict"])
model.eval()
print(f"Loaded. Teams: {list(team_encoder.classes_)}")

# --- Grid ---
xs = np.linspace(-25, 25, 25)
ys = np.linspace(0, 120, 60)
xx, yy = np.meshgrid(xs, ys)
n_points = xx.size  # 1500

def compute_grid(throw_idx: int, team_key: str, quarter_key: int) -> bytes:
    quarters_to_avg = [quarter_key] if quarter_key != 0 else [1, 2, 3, 4]
    if team_key != '':
        team_id = int(team_encoder.transform([team_key])[0])
        teams_to_avg = [team_id]
    else:
        teams_to_avg = list(range(len(team_encoder.classes_)))

    combos = len(quarters_to_avg) * len(teams_to_avg)
    all_probs = np.zeros((combos, n_points), dtype=np.float32)
    idx = 0
    with torch.no_grad():
        for q in quarters_to_avg:
            for tid in teams_to_avg:
                gp = np.column_stack([
                    xx.ravel(), yy.ravel(),
                    np.full(n_points, throw_idx),
                    np.zeros(n_points),
                    np.zeros(n_points),
                    np.full(n_points, tid),
                    np.full(n_points, q),
                ]).astype(np.float32)
                scaled = scaler.transform(gp)
                all_probs[idx] = model(torch.FloatTensor(scaled)).numpy()
                idx += 1

    grid = all_probs.mean(axis=0).reshape(60, 25).astype(np.float16)
    return grid.tobytes()

# --- Create table ---
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS epv_heatmap_cache (
        throw_idx  SMALLINT NOT NULL,
        team       VARCHAR NOT NULL DEFAULT '',
        quarter    SMALLINT NOT NULL DEFAULT 0,
        grid       BYTEA NOT NULL,
        PRIMARY KEY (throw_idx, team, quarter)
    )
""")
conn.commit()

# --- Build all combinations ---
throw_indices = list(range(1, 11))           # 1-10
team_keys = [''] + list(team_encoder.classes_)  # '' = league avg + each team
quarter_keys = [0, 1, 2, 3, 4]              # 0 = all quarters

total = len(throw_indices) * len(team_keys) * len(quarter_keys)
print(f"Total combinations: {total}")

# Check existing
cur.execute("SELECT COUNT(*) FROM epv_heatmap_cache")
existing = cur.fetchone()[0]
print(f"Existing rows: {existing} / {total}")
if existing == total:
    print("Already complete.")
    cur.close()
    conn.close()
    exit(0)

cur.execute("DELETE FROM epv_heatmap_cache")
conn.commit()

# --- Precompute ---
rows = []
t0 = time.time()
done = 0
for throw_idx in throw_indices:
    for team_key in team_keys:
        for quarter_key in quarter_keys:
            grid_bytes = compute_grid(throw_idx, team_key, quarter_key)
            rows.append((throw_idx, team_key, quarter_key, grid_bytes))
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - t0
                eta = (total - done) / (done / elapsed) / 60
                print(f"  {done}/{total} ({100*done/total:.1f}%) — ETA {eta:.1f} min")

execute_values(cur,
    "INSERT INTO epv_heatmap_cache (throw_idx, team, quarter, grid) VALUES %s ON CONFLICT DO NOTHING",
    rows, template="(%s, %s, %s, %s)"
)
conn.commit()
cur.close()
conn.close()
print(f"Done! {len(rows)} rows in {(time.time()-t0):.1f}s")
