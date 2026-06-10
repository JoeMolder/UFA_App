"""
Precompute throw density heatmaps for all players and store in Postgres.

Schema: throw_heatmap_cache(player_id, origin_xi, origin_yi, origin_x, origin_y, grid BYTEA)
grid = float16 array shape (36, 30) = 2160 bytes per row
Origins: 24x20 = 480 positions per player
Players: 244
Total: ~0.25 GB

Run: python precompute_throw_heatmap.py
Set DATABASE_URL env var to target Railway, or uses local DB by default.
"""
import os
import sys
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

# --- Model architecture (must match main.py) ---
class ContextNetwork(nn.Module):
    def __init__(self, n_players, embedding_dim=16, hidden_dim=64, output_dim=32):
        super().__init__()
        self.player_embedding = nn.Embedding(n_players, embedding_dim)
        input_dim = embedding_dim + 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, context):
        player_ids = context[:, 0].long()
        position = context[:, 1:3]
        return self.network(torch.cat([self.player_embedding(player_ids), position], dim=1))

def create_flow(n_players, num_layers=5, hidden_features=128, context_features=32):
    base_dist = distributions.StandardNormal(shape=[2])
    context_net = ContextNetwork(n_players=n_players, embedding_dim=16, hidden_dim=64, output_dim=context_features)
    tlist = []
    for _ in range(num_layers):
        tlist.append(transforms.MaskedAffineAutoregressiveTransform(
            features=2, hidden_features=hidden_features,
            context_features=context_features, num_blocks=2))
        tlist.append(transforms.ReversePermutation(features=2))
    return flows.Flow(transforms.CompositeTransform(tlist), base_dist), context_net

# --- Load model ---
MODEL_PATH = Path(__file__).resolve().parent / "models" / "normalizing_flow_model.pkl"
print(f"Loading model from {MODEL_PATH}...")
save = joblib.load(MODEL_PATH)
hp = save["hyperparameters"]
encoder = save["player_encoder"]
players = list(encoder.classes_)

flow, context_net = create_flow(
    n_players=hp["n_players"],
    num_layers=hp["num_layers"],
    hidden_features=hp["hidden_features"],
    context_features=hp["context_features"],
)
flow.load_state_dict(save["flow_state_dict"])
context_net.load_state_dict(save["context_net_state_dict"])
flow.eval()
context_net.eval()
print(f"Loaded model: {len(players)} players")

# --- Grid definitions ---
ORIGIN_X = np.linspace(-25, 25, 24)   # 24 positions
ORIGIN_Y = np.linspace(0, 120, 20)    # 20 positions
X_BINS = np.linspace(0, 1, 30)        # heatmap cols
Y_BINS = np.linspace(0, 1, 36)        # heatmap rows
xx, yy = np.meshgrid(X_BINS, Y_BINS)
GRID_POINTS = torch.FloatTensor(np.stack([xx.ravel(), yy.ravel()], axis=1))

def compute_heatmap(player_enc: int, x_norm: float, y_norm: float) -> bytes:
    ctx = torch.FloatTensor([[player_enc, x_norm, y_norm]])
    with torch.no_grad():
        ctx_feat = context_net(ctx).expand(GRID_POINTS.shape[0], -1)
        log_probs = flow.log_prob(GRID_POINTS, context=ctx_feat)
        probs = torch.exp(log_probs).numpy()
    grid_f16 = probs.reshape(len(Y_BINS), len(X_BINS)).astype(np.float16)
    return grid_f16.tobytes()

# --- Create table ---
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor()
cur.execute("""
    CREATE TABLE IF NOT EXISTS throw_heatmap_cache (
        player_id   VARCHAR NOT NULL,
        origin_xi   SMALLINT NOT NULL,
        origin_yi   SMALLINT NOT NULL,
        origin_x    REAL NOT NULL,
        origin_y    REAL NOT NULL,
        grid        BYTEA NOT NULL,
        PRIMARY KEY (player_id, origin_xi, origin_yi)
    )
""")
conn.commit()

# --- Check existing progress ---
cur.execute("SELECT COUNT(*) FROM throw_heatmap_cache")
existing = cur.fetchone()[0]
total_rows = len(players) * len(ORIGIN_X) * len(ORIGIN_Y)
print(f"Existing rows: {existing:,} / {total_rows:,}")

cur.execute("SELECT DISTINCT player_id FROM throw_heatmap_cache")
done_players = {row[0] for row in cur.fetchall()}
remaining = [p for p in players if p not in done_players]
print(f"Players remaining: {len(remaining)} / {len(players)}")

# --- Precompute ---
t0 = time.time()
for i, player in enumerate(remaining):
    player_enc = int(encoder.transform([player])[0])
    rows = []
    for xi, ox in enumerate(ORIGIN_X):
        for yi, oy in enumerate(ORIGIN_Y):
            x_norm = (float(ox) + 25) / 50
            y_norm = float(oy) / 120
            grid_bytes = compute_heatmap(player_enc, x_norm, y_norm)
            rows.append((player, xi, yi, float(ox), float(oy), grid_bytes))

    execute_values(cur,
        "INSERT INTO throw_heatmap_cache (player_id, origin_xi, origin_yi, origin_x, origin_y, grid) VALUES %s ON CONFLICT DO NOTHING",
        rows, template="(%s, %s, %s, %s, %s, %s)"
    )
    conn.commit()

    if (i + 1) % 10 == 0 or i == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(remaining) - i - 1) / rate / 60
        print(f"  {i+1}/{len(remaining)} players ({(i+1)/len(remaining)*100:.1f}%) — ETA {eta:.0f} min")

conn.close()
print(f"Done! Total time: {(time.time()-t0)/60:.1f} min")
