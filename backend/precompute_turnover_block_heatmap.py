"""
Precompute turnover and block heatmaps for all origin positions and store in Postgres.

Schema: turnover_heatmap_cache(origin_xi, origin_yi, origin_x, origin_y, grid BYTEA)
        block_heatmap_cache   (origin_xi, origin_yi, origin_x, origin_y, grid BYTEA)
grid = float16 array shape (120, 100) = 24,000 bytes per row
Origins: 24x20 = 480 positions
Total: 960 rows (~23MB)

Run: python precompute_turnover_block_heatmap.py
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
from nflows import flows, distributions, transforms

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
class TurnoverContextNetwork(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, context):
        return self.network(context)

def create_turnover_flow(num_layers=5, hidden_features=128, context_features=32):
    base_dist = distributions.StandardNormal(shape=[2])
    context_net = TurnoverContextNetwork(hidden_dim=64, output_dim=context_features)
    tlist = []
    for _ in range(num_layers):
        tlist.append(transforms.MaskedAffineAutoregressiveTransform(
            features=2, hidden_features=hidden_features,
            context_features=context_features, num_blocks=2))
        tlist.append(transforms.ReversePermutation(features=2))
    return flows.Flow(transforms.CompositeTransform(tlist), base_dist), context_net

# --- Grid definitions ---
ORIGIN_X = np.linspace(-25, 25, 24)
ORIGIN_Y = np.linspace(0, 120, 20)
X_BINS = np.linspace(0, 1, 100)
Y_BINS = np.linspace(0, 1, 120)
xx, yy = np.meshgrid(X_BINS, Y_BINS)
GRID_POINTS = torch.FloatTensor(np.stack([xx.ravel(), yy.ravel()], axis=1))

def compute_heatmap(flow, context_net, x_norm: float, y_norm: float) -> bytes:
    ctx = torch.FloatTensor([[x_norm, y_norm]])
    with torch.no_grad():
        ctx_feat = context_net(ctx).expand(GRID_POINTS.shape[0], -1)
        log_probs = flow.log_prob(GRID_POINTS, context=ctx_feat)
        probs = torch.exp(log_probs).numpy()
    grid_f16 = probs.reshape(len(Y_BINS), len(X_BINS)).astype(np.float16)
    return grid_f16.tobytes()

def precompute_for_model(model_path: Path, table_name: str, conn, cur):
    print(f"\n--- {table_name} ---")
    save = joblib.load(model_path)
    hp = save["hyperparameters"]
    flow, context_net = create_turnover_flow(
        num_layers=hp["num_layers"],
        hidden_features=hp["hidden_features"],
        context_features=hp["context_features"],
    )
    flow.load_state_dict(save["flow_state_dict"])
    context_net.load_state_dict(save["context_net_state_dict"])
    flow.eval()
    context_net.eval()
    print(f"Loaded model from {model_path}")

    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            origin_xi   SMALLINT NOT NULL,
            origin_yi   SMALLINT NOT NULL,
            origin_x    REAL NOT NULL,
            origin_y    REAL NOT NULL,
            grid        BYTEA NOT NULL,
            PRIMARY KEY (origin_xi, origin_yi)
        )
    """)
    conn.commit()

    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
    existing = cur.fetchone()[0]
    total_rows = len(ORIGIN_X) * len(ORIGIN_Y)
    print(f"Existing rows: {existing} / {total_rows}")
    if existing == total_rows:
        print("Already complete, skipping.")
        return

    cur.execute(f"DELETE FROM {table_name}")
    conn.commit()

    rows = []
    t0 = time.time()
    for xi, ox in enumerate(ORIGIN_X):
        for yi, oy in enumerate(ORIGIN_Y):
            x_norm = (float(ox) + 25) / 50
            y_norm = float(oy) / 120
            grid_bytes = compute_heatmap(flow, context_net, x_norm, y_norm)
            rows.append((xi, yi, float(ox), float(oy), grid_bytes))

    execute_values(cur,
        f"INSERT INTO {table_name} (origin_xi, origin_yi, origin_x, origin_y, grid) VALUES %s ON CONFLICT DO NOTHING",
        rows, template="(%s, %s, %s, %s, %s)"
    )
    conn.commit()
    print(f"Done: {len(rows)} rows in {time.time()-t0:.1f}s")

TURNOVER_MODEL_PATH = Path(__file__).resolve().parent / "models" / "turnover_flow_model.pkl"
BLOCK_MODEL_PATH    = Path(__file__).resolve().parent / "models" / "block_flow_model.pkl"

conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()

precompute_for_model(TURNOVER_MODEL_PATH, "turnover_heatmap_cache", conn, cur)
precompute_for_model(BLOCK_MODEL_PATH,    "block_heatmap_cache",    conn, cur)

cur.close()
conn.close()
print("\nAll done!")
