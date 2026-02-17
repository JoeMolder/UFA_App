from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path

from nflows import flows, distributions, transforms
import umap


# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="UFA Analytics API")

# CORS - allow your Vite frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection config from environment variables
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'ufa_analytics'),
    'user': os.getenv('DB_USER', 'joemolder'),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432))
}

def get_db_connection():
    """Create and return a database connection."""
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


# ---------------------------------------------------------------------------
# ML Model Loading
# ---------------------------------------------------------------------------

class ContextNetwork(nn.Module):
    """Processes context (player + position) before feeding to flow."""
    def __init__(self, n_players, embedding_dim=16, hidden_dim=64, output_dim=32):
        super().__init__()
        self.player_embedding = nn.Embedding(n_players, embedding_dim)
        input_dim = embedding_dim + 2
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, context):
        player_ids = context[:, 0].long()
        position = context[:, 1:3]
        player_emb = self.player_embedding(player_ids)
        combined = torch.cat([player_emb, position], dim=1)
        return self.network(combined)


def create_flow(n_players, num_layers=5, hidden_features=128, context_features=32):
    """Recreate flow architecture for loading weights."""
    base_dist = distributions.StandardNormal(shape=[2])
    context_net = ContextNetwork(
        n_players=n_players,
        embedding_dim=16,
        hidden_dim=64,
        output_dim=context_features,
    )
    transform_list = []
    for _ in range(num_layers):
        transform_list.append(
            transforms.MaskedAffineAutoregressiveTransform(
                features=2,
                hidden_features=hidden_features,
                context_features=context_features,
                num_blocks=2,
            )
        )
        transform_list.append(transforms.ReversePermutation(features=2))
    transform = transforms.CompositeTransform(transform_list)
    flow = flows.Flow(transform, base_dist)
    return flow, context_net


# Load model at startup
MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "normalizing_flow_model.pkl"

_flow = None
_context_net = None
_player_encoder = None
_player_names: List[str] = []
_umap_coordinates: np.ndarray | None = None
_cluster_labels: np.ndarray | None = None

try:
    save_dict = joblib.load(MODEL_PATH)
    hyperparams = save_dict["hyperparameters"]
    _player_encoder = save_dict["player_encoder"]

    _flow, _context_net = create_flow(
        n_players=hyperparams["n_players"],
        num_layers=hyperparams["num_layers"],
        hidden_features=hyperparams["hidden_features"],
        context_features=hyperparams["context_features"],
    )
    _flow.load_state_dict(save_dict["flow_state_dict"])
    _context_net.load_state_dict(save_dict["context_net_state_dict"])
    _flow.eval()
    _context_net.eval()

    _player_names = sorted(_player_encoder.classes_.tolist())
    print(f"ML model loaded from {MODEL_PATH} ({len(_player_names)} players)")
except Exception as e:
    print(f"Warning: Could not load ML model: {e}")

# ---------------------------------------------------------------------------
# Player Stats, UMAP, and Clustering (stats-based)
# ---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

_player_stats: Dict[str, Dict[str, float]] = {}
_cluster_summaries: Dict[int, Dict[str, float]] = {}

STATS_FEATURES = ['completion_pct', 'avg_throw_dist', 'avg_throw_depth', 'huck_rate', 'goal_pct', 'total_throws', 'avg_lateral_dist', 'avg_dist_from_center']

try:
    # 1. Compute per-player stats from database
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT
            thrower,
            COUNT(*) as total_throws,
            SUM(CASE WHEN event_type IN (18, 19) THEN 1 ELSE 0 END) as completions,
            SUM(CASE WHEN event_type = 19 THEN 1 ELSE 0 END) as goals,
            AVG(CASE
                WHEN event_type IN (18, 19) AND receiver_x IS NOT NULL AND receiver_y IS NOT NULL
                THEN SQRT(POWER(receiver_x - thrower_x, 2) + POWER(receiver_y - thrower_y, 2))
            END) as avg_throw_dist,
            AVG(CASE
                WHEN event_type IN (18, 19) AND receiver_y IS NOT NULL
                THEN receiver_y - thrower_y
            END) as avg_throw_depth,
            SUM(CASE
                WHEN event_type IN (18, 19) AND receiver_y IS NOT NULL
                    AND ABS(receiver_y - thrower_y) > 30
                THEN 1 ELSE 0
            END)::float / NULLIF(SUM(CASE WHEN event_type IN (18, 19) THEN 1 ELSE 0 END), 0) as huck_rate,
            AVG(CASE
                WHEN event_type IN (18, 19) AND receiver_x IS NOT NULL
                THEN ABS(receiver_x - thrower_x)
            END) as avg_lateral_dist,
            AVG(CASE
                WHEN thrower_x IS NOT NULL
                THEN ABS(thrower_x)
            END) as avg_dist_from_center
        FROM events
        WHERE thrower IS NOT NULL AND event_type IN (18, 19, 20, 22)
        GROUP BY thrower
        HAVING COUNT(*) >= 50
    """)

    for row in cur.fetchall():
        player = row['thrower']
        total = row['total_throws']
        completions = row['completions']
        _player_stats[player] = {
            'total_throws': total,
            'completion_pct': round(100 * completions / total, 1) if total > 0 else 0,
            'goal_pct': round(100 * row['goals'] / total, 1) if total > 0 else 0,
            'avg_throw_dist': round(float(row['avg_throw_dist'] or 0), 1),
            'avg_throw_depth': round(float(row['avg_throw_depth'] or 0), 1),
            'huck_rate': round(100 * float(row['huck_rate'] or 0), 1),
            'avg_lateral_dist': round(float(row['avg_lateral_dist'] or 0), 1),
            'avg_dist_from_center': round(float(row['avg_dist_from_center'] or 0), 1),
        }

    cur.close()
    conn.close()
    print(f"Player stats computed: {len(_player_stats)} players")

    # 2. Build stats feature matrix for players in the model
    if _player_encoder is not None:
        players_list = _player_encoder.classes_.tolist()
        stats_matrix = []
        for p in players_list:
            if p in _player_stats:
                s = _player_stats[p]
                stats_matrix.append([s[f] for f in STATS_FEATURES])
            else:
                stats_matrix.append([0.0] * len(STATS_FEATURES))

        stats_array = np.array(stats_matrix)

        # 3. Normalize features, then UMAP
        scaler = StandardScaler()
        stats_normalized = scaler.fit_transform(stats_array)

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.01, random_state=42)
        _umap_coordinates = reducer.fit_transform(stats_normalized)
        print(f"UMAP computed on stats features: {_umap_coordinates.shape}")

        # 4. K-Means clustering on normalized stats
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
        _cluster_labels = kmeans.fit_predict(stats_normalized)
        print(f"K-Means: 6 clusters, sizes: {[int((_cluster_labels == i).sum()) for i in range(6)]}")

        # 5. Compute cluster summaries
        for cluster_id in set(_cluster_labels.tolist()):
            cluster_players = [
                players_list[i] for i, c in enumerate(_cluster_labels) if c == cluster_id
            ]
            stats_in_cluster = [_player_stats[p] for p in cluster_players if p in _player_stats]
            if stats_in_cluster:
                _cluster_summaries[cluster_id] = {
                    'count': len(cluster_players),
                    'avg_completion_pct': round(np.mean([s['completion_pct'] for s in stats_in_cluster]), 1),
                    'avg_throw_dist': round(np.mean([s['avg_throw_dist'] for s in stats_in_cluster]), 1),
                    'avg_throw_depth': round(np.mean([s['avg_throw_depth'] for s in stats_in_cluster]), 1),
                    'avg_huck_rate': round(np.mean([s['huck_rate'] for s in stats_in_cluster]), 1),
                    'avg_goal_pct': round(np.mean([s['goal_pct'] for s in stats_in_cluster]), 1),
                    'avg_total_throws': round(np.mean([s['total_throws'] for s in stats_in_cluster]), 0),
                    'avg_lateral_dist': round(np.mean([s['avg_lateral_dist'] for s in stats_in_cluster]), 1),
                    'avg_dist_from_center': round(np.mean([s['avg_dist_from_center'] for s in stats_in_cluster]), 1),
                }
        print(f"Cluster summaries: {len(_cluster_summaries)} clusters")
except Exception as e:
    print(f"Warning: Could not compute stats/clusters: {e}")


# ---------------------------------------------------------------------------
# Database Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": "UFA Analytics API",
        "status": "running",
        "version": "1.0.0",
        "model_loaded": _flow is not None,
    }


@app.get("/games")
def get_games(limit: int = 10) -> List[Dict[str, Any]]:
    """Get list of games from database."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT
                game_id,
                home_team_id,
                away_team_id,
                home_score,
                away_score,
                game_date,
                year
            FROM games
            ORDER BY game_date DESC
            LIMIT %s;
        """, (limit,))

        games = cur.fetchall()

        cur.close()
        conn.close()

        return games

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/games/{game_id}")
def get_game(game_id: str) -> Dict[str, Any]:
    """Get details for a specific game."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Get game info
        cur.execute("""
            SELECT
                game_id,
                home_team_id,
                away_team_id,
                home_score,
                away_score,
                game_date,
                year
            FROM games
            WHERE game_id = %s;
        """, (game_id,))

        game = cur.fetchone()

        if not game:
            raise HTTPException(status_code=404, detail="Game not found")

        # Get event counts
        cur.execute("""
            SELECT
                event_type,
                COUNT(*) as count
            FROM events
            WHERE game_id = %s
            GROUP BY event_type;
        """, (game_id,))

        event_counts = {row['event_type']: row['count'] for row in cur.fetchall()}

        cur.close()
        conn.close()

        return {
            **game,
            "event_counts": event_counts
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/games/{game_id}/events")
def get_game_events(game_id: str) -> List[Dict[str, Any]]:
    """Get all events for a specific game."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT
                event_id,
                event_number,
                event_type,
                team,
                year,
                time,
                puller,
                pull_x,
                pull_y,
                thrower,
                thrower_x,
                thrower_y,
                receiver,
                receiver_x,
                receiver_y,
                turnover_x,
                turnover_y,
                defender,
                synthetic
            FROM events
            WHERE game_id = %s
            ORDER BY event_number;
        """, (game_id,))

        events = cur.fetchall()

        cur.close()
        conn.close()

        return events

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/summary")
def get_stats_summary() -> Dict[str, Any]:
    """Get overall database statistics."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Total games
        cur.execute("SELECT COUNT(*) as count FROM games;")
        total_games = cur.fetchone()['count']

        # Total events
        cur.execute("SELECT COUNT(*) as count FROM events;")
        total_events = cur.fetchone()['count']

        # Synthetic events
        cur.execute("SELECT COUNT(*) as count FROM events WHERE synthetic = TRUE;")
        synthetic_events = cur.fetchone()['count']

        cur.close()
        conn.close()

        return {
            "total_games": total_games,
            "total_events": total_events,
            "synthetic_events": synthetic_events
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# ML Prediction Endpoints
# ---------------------------------------------------------------------------

@app.get("/players")
def get_players() -> List[str]:
    """Get list of player names available for throw prediction."""
    if not _player_names:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _player_names


@app.get("/embeddings/players")
def get_player_embeddings() -> Dict[str, Any]:
    """Get UMAP 2D projection of player embeddings."""
    if _umap_coordinates is None or _player_encoder is None or _cluster_labels is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Return in label encoder order (matches embedding matrix row order)
    players = _player_encoder.classes_.tolist()
    # Per-player stats (keyed by name)
    player_stats = {p: _player_stats[p] for p in players if p in _player_stats}
    # Cluster summaries (keyed by cluster id as string)
    cluster_info = {str(k): v for k, v in _cluster_summaries.items()}
    return {
        "players": players,
        "coordinates": _umap_coordinates.tolist(),
        "clusters": _cluster_labels.tolist(),
        "player_stats": player_stats,
        "cluster_summaries": cluster_info,
    }


@app.get("/predict/throws")
def predict_throws(
    player: str = Query(..., description="Player ID"),
    x: float = Query(..., description="Thrower X position (-25 to 25)"),
    y: float = Query(..., description="Thrower Y position (0 to 120)"),
    grid_size: int = Query(30, ge=10, le=200, description="Grid resolution"),
) -> Dict[str, Any]:
    """Predict throw distribution for a player at a field position."""
    if _flow is None or _context_net is None or _player_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate player
    if player not in _player_encoder.classes_:
        raise HTTPException(status_code=404, detail=f"Player '{player}' not found in model")

    # Encode player and normalize coords
    player_encoded = int(_player_encoder.transform([player])[0])
    x_norm = (x + 25) / 50
    y_norm = y / 120

    context_input = torch.FloatTensor([[player_encoded, x_norm, y_norm]])

    with torch.no_grad():
        context_features = _context_net(context_input)

        x_bins = np.linspace(0, 1, grid_size)
        y_bins = np.linspace(0, 1, int(grid_size * 1.2))

        xx, yy = np.meshgrid(x_bins, y_bins)
        grid_points = torch.FloatTensor(np.stack([xx.ravel(), yy.ravel()], axis=1))

        context_expanded = context_features.expand(grid_points.shape[0], -1)
        log_probs = _flow.log_prob(grid_points, context=context_expanded)
        probs = torch.exp(log_probs).numpy()

        grid = probs.reshape(len(y_bins), len(x_bins))

    return {
        "grid": grid.tolist(),
        "extent": [-25, 25, 0, 120],
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
