from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import numpy as np
import torch
import torch.nn as nn
import joblib
from pathlib import Path
from scipy.ndimage import gaussian_filter

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


class TurnoverContextNetwork(nn.Module):
    """Processes thrower position context (no player embedding)."""
    def __init__(self, hidden_dim=64, output_dim=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, context):
        return self.network(context)


def create_turnover_flow(num_layers=5, hidden_features=128, context_features=32):
    """Recreate turnover flow architecture for loading weights."""
    base_dist = distributions.StandardNormal(shape=[2])
    context_net = TurnoverContextNetwork(hidden_dim=64, output_dim=context_features)
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

# Load turnover flow model
TURNOVER_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "turnover_flow_model.pkl"

_turnover_flow = None
_turnover_context_net = None

try:
    turnover_save = joblib.load(TURNOVER_MODEL_PATH)
    turnover_hp = turnover_save["hyperparameters"]

    _turnover_flow, _turnover_context_net = create_turnover_flow(
        num_layers=turnover_hp["num_layers"],
        hidden_features=turnover_hp["hidden_features"],
        context_features=turnover_hp["context_features"],
    )
    _turnover_flow.load_state_dict(turnover_save["flow_state_dict"])
    _turnover_context_net.load_state_dict(turnover_save["context_net_state_dict"])
    _turnover_flow.eval()
    _turnover_context_net.eval()
    print(f"Turnover flow model loaded from {TURNOVER_MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load turnover flow model: {e}")

# Load block flow model (throwaways preceding blocks)
BLOCK_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "block_flow_model.pkl"

_block_flow = None
_block_context_net = None

try:
    block_save = joblib.load(BLOCK_MODEL_PATH)
    block_hp = block_save["hyperparameters"]

    # Same architecture as turnover flow (TurnoverContextNetwork with 2 inputs)
    _block_flow, _block_context_net = create_turnover_flow(
        num_layers=block_hp["num_layers"],
        hidden_features=block_hp["hidden_features"],
        context_features=block_hp["context_features"],
    )
    _block_flow.load_state_dict(block_save["flow_state_dict"])
    _block_context_net.load_state_dict(block_save["context_net_state_dict"])
    _block_flow.eval()
    _block_context_net.eval()
    print(f"Block flow model loaded from {BLOCK_MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load block flow model: {e}")

# Load completion flow model (for relative density computation)
COMPLETION_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "completion_flow_model.pkl"

_completion_flow = None
_completion_context_net = None

try:
    completion_save = joblib.load(COMPLETION_MODEL_PATH)
    completion_hp = completion_save["hyperparameters"]

    _completion_flow, _completion_context_net = create_turnover_flow(
        num_layers=completion_hp["num_layers"],
        hidden_features=completion_hp["hidden_features"],
        context_features=completion_hp["context_features"],
    )
    _completion_flow.load_state_dict(completion_save["flow_state_dict"])
    _completion_context_net.load_state_dict(completion_save["context_net_state_dict"])
    _completion_flow.eval()
    _completion_context_net.eval()
    print(f"Completion flow model loaded from {COMPLETION_MODEL_PATH}")
except Exception as e:
    print(f"Warning: Could not load completion flow model: {e}")

# Load pull play CVAE model
class PullPlayCVAE(nn.Module):
    """Conditional VAE for pull play sequences (with z-skip decoder)."""
    def __init__(self, n_teams, seq_dim=12, latent_dim=16,
                 team_embed_dim=8, condition_dim=16, hidden_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_dim = seq_dim
        self.team_embedding = nn.Embedding(n_teams + 1, team_embed_dim)
        self.condition_net = nn.Sequential(
            nn.Linear(2 + team_embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, condition_dim),
        )
        self.encoder = nn.Sequential(
            nn.Linear(seq_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        # Decoder with z injected at two layers to prevent posterior collapse
        self.decoder_fc1 = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder_fc2 = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder_out = nn.Sequential(
            nn.Linear(hidden_dim, seq_dim),
            nn.Sigmoid(),
        )

    def get_condition(self, pull_pos, team_ids):
        team_emb = self.team_embedding(team_ids)
        combined = torch.cat([pull_pos, team_emb], dim=1)
        return self.condition_net(combined)

    def decode(self, z, condition):
        h = self.decoder_fc1(torch.cat([z, condition], dim=1))
        h = self.decoder_fc2(torch.cat([h, z], dim=1))
        return self.decoder_out(h)

    def sample(self, pull_pos, team_ids, n_samples=1):
        self.eval()
        with torch.no_grad():
            condition = self.get_condition(pull_pos, team_ids)
            condition = condition.repeat(n_samples, 1)
            z = torch.randn(n_samples, self.latent_dim)
            return self.decode(z, condition)


CVAE_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "pull_play_cvae.pkl"

_cvae_model: Optional[PullPlayCVAE] = None
_cvae_team_encoder = None
_cvae_cluster_centers: Optional[np.ndarray] = None  # [K, latent_dim] latent centers
_cvae_cluster_counts: Optional[np.ndarray] = None   # [K] sequence counts

try:
    cvae_save = joblib.load(CVAE_MODEL_PATH)
    cvae_hp = cvae_save["hyperparameters"]
    _cvae_team_encoder = cvae_save["team_encoder"]

    _cvae_model = PullPlayCVAE(
        n_teams=cvae_save["n_real_teams"],
        seq_dim=cvae_hp["seq_dim"],
        latent_dim=cvae_hp["latent_dim"],
        team_embed_dim=cvae_hp["team_embed_dim"],
        condition_dim=cvae_hp["condition_dim"],
        hidden_dim=cvae_hp["hidden_dim"],
    )
    _cvae_model.load_state_dict(cvae_save["model_state_dict"])
    _cvae_model.eval()

    if "kmeans" in cvae_save:
        _cvae_cluster_centers = cvae_save["kmeans"].cluster_centers_  # [K, latent_dim]
        _cvae_cluster_counts = cvae_save["cluster_counts"]
        print(f"Pull play CVAE loaded ({len(_cvae_cluster_centers)} cluster archetypes)")
    else:
        print("Pull play CVAE loaded (no cluster data)")
except Exception as e:
    print(f"Warning: Could not load pull play CVAE: {e}")


# Load EPV (Expected Possession Value) models
class EPVNet(nn.Module):
    """Small neural net for EPV prediction. Must match training notebook architecture."""
    def __init__(self, input_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


EPV_XGB_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "epv_xgb.pkl"
EPV_NN_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "epv_nn.pkl"

_epv_xgb = None
_epv_nn: Optional[EPVNet] = None
_epv_scaler = None
_epv_team_encoder = None

try:
    epv_xgb_save = joblib.load(EPV_XGB_PATH)
    _epv_xgb = epv_xgb_save["model"]
    _epv_team_encoder = epv_xgb_save["team_encoder"]
    print(f"EPV XGBoost loaded (AUC={epv_xgb_save['metrics']['auc']:.4f})")
except Exception as e:
    print(f"Warning: Could not load EPV XGBoost model: {e}")

try:
    epv_nn_save = joblib.load(EPV_NN_PATH)
    _epv_scaler = epv_nn_save["scaler"]
    if _epv_team_encoder is None:
        _epv_team_encoder = epv_nn_save["team_encoder"]
    _epv_nn = EPVNet(input_dim=epv_nn_save.get("input_dim", 6))
    _epv_nn.load_state_dict(epv_nn_save["model_state_dict"])
    _epv_nn.eval()
    print(f"EPV Neural Net loaded (AUC={epv_nn_save['metrics']['auc']:.4f})")
except Exception as e:
    print(f"Warning: Could not load EPV neural net model: {e}")

# Load completion XGBoost model (per-thrower P(completion | from, to))
COMPLETION_XGB_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "completion_xgb.pkl"

_completion_xgb = None
_completion_thrower_encoder = None

try:
    completion_xgb_save = joblib.load(COMPLETION_XGB_PATH)
    _completion_xgb = completion_xgb_save["model"]
    _completion_thrower_encoder = completion_xgb_save["encoder"]
    print(f"Completion XGBoost loaded (AUC={completion_xgb_save['metrics']['auc']:.4f}, {len(_completion_thrower_encoder.classes_)} throwers)")
except Exception as e:
    print(f"Warning: Could not load completion XGBoost model: {e}")

# ---------------------------------------------------------------------------
# Player Stats, UMAP, and Clustering (stats-based)
# ---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

_player_stats: Dict[str, Dict[str, float]] = {}
_cluster_summaries: Dict[int, Dict[str, float]] = {}

STATS_FEATURES = ['completion_pct', 'avg_throw_dist', 'avg_throw_depth', 'huck_rate', 'goal_pct', 'avg_lateral_dist', 'avg_dist_from_center']

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
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        _cluster_labels = kmeans.fit_predict(stats_normalized)
        print(f"K-Means: 4 clusters, sizes: {[int((_cluster_labels == i).sum()) for i in range(4)]}")

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
def get_games(limit: int = 10, search: str = Query(default="")) -> List[Dict[str, Any]]:
    """Get list of games from database. Optional search by team or game ID."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if search.strip():
            pattern = f"%{search.strip().lower()}%"
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
                WHERE LOWER(game_id) LIKE %s
                   OR LOWER(home_team_id) LIKE %s
                   OR LOWER(away_team_id) LIKE %s
                ORDER BY game_date DESC
                LIMIT %s;
            """, (pattern, pattern, pattern, limit))
        else:
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


@app.get("/teams")
def get_teams() -> List[str]:
    """Get list of all team names."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT team FROM events ORDER BY team")
    teams = [row['team'] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return teams


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


@app.get("/predict/throws/batch")
def predict_throws_batch(
    player: str = Query(..., description="Player ID"),
    grid_cells_x: int = Query(10, ge=3, le=20, description="Number of grid cells across field width"),
    grid_cells_y: int = Query(12, ge=3, le=24, description="Number of grid cells across field length"),
    heatmap_resolution: int = Query(30, ge=10, le=200, description="Resolution of each heatmap"),
) -> Dict[str, Any]:
    """Pre-compute throw predictions for a grid of thrower positions."""
    if _flow is None or _context_net is None or _player_encoder is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if player not in _player_encoder.classes_:
        raise HTTPException(status_code=404, detail=f"Player '{player}' not found in model")

    player_encoded = int(_player_encoder.transform([player])[0])

    # Build grid of thrower positions across the field
    x_positions = np.linspace(-25, 25, grid_cells_x)
    y_positions = np.linspace(0, 120, grid_cells_y)

    # Pre-compute the heatmap grid points (shared across all positions)
    x_bins = np.linspace(0, 1, heatmap_resolution)
    y_bins = np.linspace(0, 1, int(heatmap_resolution * 1.2))
    xx, yy = np.meshgrid(x_bins, y_bins)
    grid_points = torch.FloatTensor(np.stack([xx.ravel(), yy.ravel()], axis=1))

    results = {}

    with torch.no_grad():
        for xi, field_x in enumerate(x_positions):
            for yi, field_y in enumerate(y_positions):
                x_norm = (field_x + 25) / 50
                y_norm = field_y / 120

                context_input = torch.FloatTensor([[player_encoded, x_norm, y_norm]])
                context_features = _context_net(context_input)
                context_expanded = context_features.expand(grid_points.shape[0], -1)
                log_probs = _flow.log_prob(grid_points, context=context_expanded)
                probs = torch.exp(log_probs).numpy()
                grid = probs.reshape(len(y_bins), len(x_bins))

                results[f"{xi},{yi}"] = grid.tolist()

    return {
        "grids": results,
        "x_positions": x_positions.tolist(),
        "y_positions": y_positions.tolist(),
        "extent": [-25, 25, 0, 120],
    }


@app.get("/predict/turnovers")
def predict_turnovers(
    x: float = Query(..., description="Thrower X position (-25 to 25)"),
    y: float = Query(..., description="Thrower Y position (0 to 120)"),
    grid_size: int = Query(30, ge=10, le=200, description="Grid resolution"),
) -> Dict[str, Any]:
    """Predict turnover destination distribution from a field position."""
    if _turnover_flow is None or _turnover_context_net is None:
        raise HTTPException(status_code=503, detail="Turnover model not loaded")

    x_norm = (x + 25) / 50
    y_norm = y / 120

    context_input = torch.FloatTensor([[x_norm, y_norm]])

    with torch.no_grad():
        context_features = _turnover_context_net(context_input)

        x_bins = np.linspace(0, 1, grid_size)
        y_bins = np.linspace(0, 1, int(grid_size * 1.2))

        xx, yy = np.meshgrid(x_bins, y_bins)
        grid_points = torch.FloatTensor(np.stack([xx.ravel(), yy.ravel()], axis=1))

        context_expanded = context_features.expand(grid_points.shape[0], -1)
        log_probs = _turnover_flow.log_prob(grid_points, context=context_expanded)
        probs = torch.exp(log_probs).numpy()

        grid = probs.reshape(len(y_bins), len(x_bins))

    return {
        "grid": grid.tolist(),
        "extent": [-25, 25, 0, 120],
    }


@app.get("/predict/relative-density")
def predict_relative_density(
    x: float = Query(..., description="Thrower X position (-25 to 25)"),
    y: float = Query(..., description="Thrower Y position (0 to 120)"),
    grid_size: int = Query(30, ge=10, le=200, description="Grid resolution"),
) -> Dict[str, Any]:
    """Compute turnover/completion density ratio from a field position.

    High values = turnovers land here disproportionately vs completions.
    """
    if _turnover_flow is None or _turnover_context_net is None:
        raise HTTPException(status_code=503, detail="Turnover model not loaded")
    if _completion_flow is None or _completion_context_net is None:
        raise HTTPException(status_code=503, detail="Completion model not loaded")

    x_norm = (x + 25) / 50
    y_norm = y / 120
    context_input = torch.FloatTensor([[x_norm, y_norm]])

    with torch.no_grad():
        x_bins = np.linspace(0, 1, grid_size)
        y_bins = np.linspace(0, 1, int(grid_size * 1.2))
        xx, yy = np.meshgrid(x_bins, y_bins)
        grid_points = torch.FloatTensor(np.stack([xx.ravel(), yy.ravel()], axis=1))

        # Turnover density
        turnover_ctx = _turnover_context_net(context_input)
        turnover_expanded = turnover_ctx.expand(grid_points.shape[0], -1)
        turnover_log_probs = _turnover_flow.log_prob(grid_points, context=turnover_expanded)
        turnover_probs = torch.exp(turnover_log_probs).numpy()

        # Completion density
        completion_ctx = _completion_context_net(context_input)
        completion_expanded = completion_ctx.expand(grid_points.shape[0], -1)
        completion_log_probs = _completion_flow.log_prob(grid_points, context=completion_expanded)
        completion_probs = torch.exp(completion_log_probs).numpy()

        # Ratio: turnover / completion (with small epsilon to avoid division by zero)
        epsilon = 1e-8
        ratio = turnover_probs / (completion_probs + epsilon)
        grid = ratio.reshape(len(y_bins), len(x_bins))

    return {
        "grid": grid.tolist(),
        "extent": [-25, 25, 0, 120],
    }


@app.get("/predict/blocks")
def predict_blocks(
    x: float = Query(..., description="Thrower X position (-25 to 25)"),
    y: float = Query(..., description="Thrower Y position (0 to 120)"),
    grid_size: int = Query(30, ge=10, le=200, description="Grid resolution"),
) -> Dict[str, Any]:
    """Predict block destination distribution from a field position."""
    if _block_flow is None or _block_context_net is None:
        raise HTTPException(status_code=503, detail="Block model not loaded")

    x_norm = (x + 25) / 50
    y_norm = y / 120

    context_input = torch.FloatTensor([[x_norm, y_norm]])

    with torch.no_grad():
        context_features = _block_context_net(context_input)

        x_bins = np.linspace(0, 1, grid_size)
        y_bins = np.linspace(0, 1, int(grid_size * 1.2))

        xx, yy = np.meshgrid(x_bins, y_bins)
        grid_points = torch.FloatTensor(np.stack([xx.ravel(), yy.ravel()], axis=1))

        context_expanded = context_features.expand(grid_points.shape[0], -1)
        log_probs = _block_flow.log_prob(grid_points, context=context_expanded)
        probs = torch.exp(log_probs).numpy()

        grid = probs.reshape(len(y_bins), len(x_bins))

    return {
        "grid": grid.tolist(),
        "extent": [-25, 25, 0, 120],
    }


@app.get("/heatmap/turnovers")
def get_turnover_heatmap(
    grid_x: int = Query(50, ge=10, le=200, description="Grid bins across field width"),
    grid_y: int = Query(60, ge=10, le=200, description="Grid bins across field length"),
    smooth: float = Query(2.0, ge=0, le=10, description="Gaussian smoothing sigma"),
) -> Dict[str, Any]:
    """Return throw volume and turnover grids for the entire dataset."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # 1. All throw destinations (completions + goals + drops)
        cur.execute("""
            SELECT receiver_x, receiver_y
            FROM events
            WHERE event_type IN (18, 19, 20)
              AND receiver_x IS NOT NULL
              AND receiver_y IS NOT NULL
        """)
        throw_rows = cur.fetchall()

        # 2. Turnover destinations: drops + throwaways
        cur.execute("""
            SELECT receiver_x AS x, receiver_y AS y FROM events
            WHERE event_type = 20 AND receiver_x IS NOT NULL AND receiver_y IS NOT NULL
            UNION ALL
            SELECT turnover_x AS x, turnover_y AS y FROM events
            WHERE event_type = 22 AND turnover_x IS NOT NULL AND turnover_y IS NOT NULL
        """)
        turnover_rows = cur.fetchall()

        cur.close()
        conn.close()

        # Bin into grids
        x_edges = np.linspace(-25, 25, grid_x + 1)
        y_edges = np.linspace(0, 120, grid_y + 1)

        throw_grid = np.zeros((grid_y, grid_x), dtype=np.float64)
        for row in throw_rows:
            xi = np.searchsorted(x_edges, float(row['receiver_x']), side='right') - 1
            yi = np.searchsorted(y_edges, float(row['receiver_y']), side='right') - 1
            if 0 <= xi < grid_x and 0 <= yi < grid_y:
                throw_grid[yi, xi] += 1

        turnover_grid = np.zeros((grid_y, grid_x), dtype=np.float64)
        for row in turnover_rows:
            xi = np.searchsorted(x_edges, float(row['x']), side='right') - 1
            yi = np.searchsorted(y_edges, float(row['y']), side='right') - 1
            if 0 <= xi < grid_x and 0 <= yi < grid_y:
                turnover_grid[yi, xi] += 1

        # Optional Gaussian smoothing
        if smooth > 0:
            throw_grid = gaussian_filter(throw_grid, sigma=smooth)
            turnover_grid = gaussian_filter(turnover_grid, sigma=smooth)

        return {
            "throw_grid": throw_grid.tolist(),
            "turnover_grid": turnover_grid.tolist(),
            "total_throws": int(len(throw_rows)),
            "total_turnovers": int(len(turnover_rows)),
            "extent": [-25, 25, 0, 120],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/heatmap/throwaways/batch")
def get_throwaway_heatmap_batch(
    thrower_grid_x: int = Query(10, ge=3, le=20, description="Number of thrower positions across width"),
    thrower_grid_y: int = Query(24, ge=3, le=40, description="Number of thrower positions across length"),
    dest_grid_x: int = Query(30, ge=5, le=300, description="Destination grid bins (width)"),
    dest_grid_y: int = Query(36, ge=5, le=300, description="Destination grid bins (length)"),
    radius: float = Query(12, ge=3, le=40, description="Gaussian kernel radius for thrower weighting"),
) -> Dict[str, Any]:
    """Batch compute throwaway destination density grids for all thrower positions.

    Shows where throwaways land (turnover_x/y) given the thrower position.
    Direction-normalized so all throws attack toward y=110.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Throwaways that are followed by a block (type 11)
        cur.execute("""
            SELECT
                CASE WHEN e.team = g.away_team_id THEN -e.thrower_x ELSE e.thrower_x END as thrower_x,
                CASE WHEN e.team = g.away_team_id THEN 120 - e.thrower_y ELSE e.thrower_y END as thrower_y,
                CASE WHEN e.team = g.away_team_id THEN -e.turnover_x ELSE e.turnover_x END as dest_x,
                CASE WHEN e.team = g.away_team_id THEN 120 - e.turnover_y ELSE e.turnover_y END as dest_y
            FROM events e
            JOIN events nxt
                ON nxt.game_id = e.game_id
                AND nxt.event_number = e.event_number + 1
            JOIN games g ON e.game_id = g.game_id
            WHERE e.event_type = 22
              AND nxt.event_type = 11
              AND e.thrower_x IS NOT NULL AND e.thrower_y IS NOT NULL
              AND e.turnover_x IS NOT NULL AND e.turnover_y IS NOT NULL
        """)
        rows = cur.fetchall()

        cur.close()
        conn.close()

        if not rows:
            raise HTTPException(status_code=404, detail="No throwaway data found")

        # Convert to numpy arrays
        thrower_xy = np.array([[float(r['thrower_x']), float(r['thrower_y'])] for r in rows])
        dest_xy = np.array([[float(r['dest_x']), float(r['dest_y'])] for r in rows])

        # Pre-bin destinations
        x_edges = np.linspace(-25, 25, dest_grid_x + 1)
        y_edges = np.linspace(0, 120, dest_grid_y + 1)
        dest_xi = np.clip(np.searchsorted(x_edges, dest_xy[:, 0], side='right') - 1, 0, dest_grid_x - 1)
        dest_yi = np.clip(np.searchsorted(y_edges, dest_xy[:, 1], side='right') - 1, 0, dest_grid_y - 1)

        # Thrower grid positions
        x_positions = np.linspace(-25, 25, thrower_grid_x).tolist()
        y_positions = np.linspace(0, 120, thrower_grid_y).tolist()

        two_sigma_sq = 2 * radius * radius
        grids: Dict[str, list] = {}

        for xi, qx in enumerate(x_positions):
            for yi, qy in enumerate(y_positions):
                dist_sq = (thrower_xy[:, 0] - qx) ** 2 + (thrower_xy[:, 1] - qy) ** 2
                weights = np.exp(-dist_sq / two_sigma_sq)

                block_grid = np.zeros((dest_grid_y, dest_grid_x), dtype=np.float64)
                np.add.at(block_grid, (dest_yi, dest_xi), weights)

                # Normalize to probability distribution
                total = block_grid.sum()
                if total > 0:
                    block_grid /= total

                block_grid = gaussian_filter(block_grid, sigma=0.8)
                grids[f"{xi},{yi}"] = block_grid.tolist()

        return {
            "grids": grids,
            "x_positions": x_positions,
            "y_positions": y_positions,
            "extent": [-25, 25, 0, 120],
            "total_throwaways": len(rows),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/heatmap/turnover-origins")
def get_turnover_origins(
    player: Optional[str] = Query(None, description="Player name to filter by (omit for all players)"),
    team: Optional[str] = Query(None, description="Team name to filter by (throwing team)"),
    opponent: Optional[str] = Query(None, description="Opponent team to filter by"),
    grid_x: int = Query(50, ge=10, le=200, description="Grid bins across field width"),
    grid_y: int = Query(60, ge=10, le=200, description="Grid bins across field length"),
    smooth: float = Query(2.0, ge=0, le=10, description="Gaussian smoothing sigma"),
) -> Dict[str, Any]:
    """Return turnover rate heatmap by thrower position (turnovers / total throws)."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Build dynamic filters
        filters = []
        params: list = []
        if player:
            filters.append("AND e.thrower = %s")
            params.append(player)
        if team:
            filters.append("AND e.team = %s")
            params.append(team)
        if opponent:
            filters.append("AND CASE WHEN e.team = g.home_team_id THEN g.away_team_id ELSE g.home_team_id END = %s")
            params.append(opponent)
        filter_sql = " ".join(filters)

        # All throws (completions 18, goals 19, drops 20, throwaways 22),
        # direction-normalized so all throws attack toward y=110.
        # Filter to playing field only (y between 0 and 100, excludes attacking end zone).
        cur.execute(f"""
            SELECT x, y FROM (
                SELECT
                    CASE WHEN e.team = g.away_team_id THEN -e.thrower_x ELSE e.thrower_x END as x,
                    CASE WHEN e.team = g.away_team_id THEN 120 - e.thrower_y ELSE e.thrower_y END as y
                FROM events e
                JOIN games g ON e.game_id = g.game_id
                WHERE e.event_type IN (18, 19, 20, 22)
                  AND e.thrower_x IS NOT NULL
                  AND e.thrower_y IS NOT NULL
                  {filter_sql}
            ) sub
            WHERE y >= 0 AND y <= 100
        """, params)
        all_throw_rows = cur.fetchall()

        # Turnover throws only (drops 20, throwaways 22)
        cur.execute(f"""
            SELECT x, y FROM (
                SELECT
                    CASE WHEN e.team = g.away_team_id THEN -e.thrower_x ELSE e.thrower_x END as x,
                    CASE WHEN e.team = g.away_team_id THEN 120 - e.thrower_y ELSE e.thrower_y END as y
                FROM events e
                JOIN games g ON e.game_id = g.game_id
                WHERE e.event_type IN (20, 22)
                  AND e.thrower_x IS NOT NULL
                  AND e.thrower_y IS NOT NULL
                  {filter_sql}
            ) sub
            WHERE y >= 0 AND y <= 100
        """, params)
        turnover_rows = cur.fetchall()

        cur.close()
        conn.close()

        if not all_throw_rows:
            raise HTTPException(status_code=404, detail="No throw data found")

        x_edges = np.linspace(-25, 25, grid_x + 1)
        y_edges = np.linspace(0, 120, grid_y + 1)

        throw_grid = np.zeros((grid_y, grid_x), dtype=np.float64)
        for row in all_throw_rows:
            xi = np.searchsorted(x_edges, float(row['x']), side='right') - 1
            yi = np.searchsorted(y_edges, float(row['y']), side='right') - 1
            if 0 <= xi < grid_x and 0 <= yi < grid_y:
                throw_grid[yi, xi] += 1

        turnover_grid = np.zeros((grid_y, grid_x), dtype=np.float64)
        for row in turnover_rows:
            xi = np.searchsorted(x_edges, float(row['x']), side='right') - 1
            yi = np.searchsorted(y_edges, float(row['y']), side='right') - 1
            if 0 <= xi < grid_x and 0 <= yi < grid_y:
                turnover_grid[yi, xi] += 1

        # Turnover rate from raw counts (before smoothing to avoid bleeding into empty cells)
        rate_grid = np.zeros_like(throw_grid)
        mask = throw_grid > 0
        rate_grid[mask] = turnover_grid[mask] / throw_grid[mask]

        # Smooth the rate grid (not the raw counts)
        if smooth > 0:
            rate_grid = gaussian_filter(rate_grid, sigma=smooth)

        # Zero out cells beyond the playing field (y > 100) after smoothing
        # to prevent bleed into the attacking end zone
        y_100_idx = np.searchsorted(y_edges, 100, side='right') - 1
        if y_100_idx < grid_y:
            rate_grid[y_100_idx + 1:, :] = 0

        return {
            "grid": rate_grid.tolist(),
            "total_throws": len(all_throw_rows),
            "total_turnovers": len(turnover_rows),
            "extent": [-25, 25, 0, 120],
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _cvae_predict(pull_x: float, pull_y: float, team: Optional[str], n_samples: int = 1) -> List[Dict]:
    """Use CVAE to generate throw sequence(s). Returns list of sequences."""
    if _cvae_model is None or _cvae_team_encoder is None:
        raise HTTPException(status_code=503, detail="CVAE model not loaded")

    pull_x_norm = (pull_x + 25) / 50
    pull_y_norm = pull_y / 120
    pull_tensor = torch.FloatTensor([[pull_x_norm, pull_y_norm]])

    if team and team in _cvae_team_encoder.classes_:
        team_id = int(_cvae_team_encoder.transform([team])[0]) + 1
    else:
        team_id = 0  # all-teams token
    team_tensor = torch.LongTensor([team_id])

    samples = _cvae_model.sample(pull_tensor, team_tensor, n_samples=n_samples).numpy()

    def denorm(seq):
        # x coords at indices 0,2,4,6,8,10: [0,1] -> [-25,25]
        # y coords at indices 1,3,5,7,9,11: [0,1] -> [0,120]
        result = seq.copy()
        for i in [0, 2, 4, 6, 8, 10]:
            result[i] = float(result[i]) * 50 - 25
        for i in [1, 3, 5, 7, 9, 11]:
            result[i] = float(result[i]) * 120
        return result

    sequences = []
    for s in range(n_samples):
        seq = denorm(samples[s])
        sequences.append([
            {"from_x": round(float(seq[0]), 1), "from_y": round(float(seq[1]), 1),
             "to_x": round(float(seq[2]), 1), "to_y": round(float(seq[3]), 1)},
            {"from_x": round(float(seq[4]), 1), "from_y": round(float(seq[5]), 1),
             "to_x": round(float(seq[6]), 1), "to_y": round(float(seq[7]), 1)},
            {"from_x": round(float(seq[8]), 1), "from_y": round(float(seq[9]), 1),
             "to_x": round(float(seq[10]), 1), "to_y": round(float(seq[11]), 1)},
        ])
    return sequences


@app.get("/pull-play/sequence")
def get_pull_play_sequence(
    pull_x: float = Query(0.0, description="Pull landing X position (-25 to 25)"),
    pull_y: float = Query(90.0, description="Pull landing Y position (0 to 120)"),
    team: Optional[str] = Query(None, description="Receiving team (omit for all teams)"),
    radius: float = Query(15.0, ge=1, le=50, description="Search radius in yards around pull landing"),
    mode: str = Query("model", description="'model' for CVAE prediction, 'average' for data average"),
) -> Dict[str, Any]:
    """Return first 3 throws after a pull landing using CVAE model or data average."""
    try:
        # CVAE model mode
        if mode == "model":
            sequences = _cvae_predict(pull_x, pull_y, team, n_samples=1)
            return {
                "throws": sequences[0],
                "sample_size": None,
                "pull_landing": {"x": pull_x, "y": pull_y},
                "scoring_rate": None,
                "mode": "model",
            }

        # Data average mode
        conn = get_db_connection()
        cur = conn.cursor()

        team_filter = "AND recv_team = %s" if team else ""
        params: list = [pull_x, pull_y, radius]
        if team:
            params.append(team)

        cur.execute(f"""
            WITH pull_data AS (
                SELECT
                    p.game_id, p.event_number AS pull_num,
                    CASE WHEN recv.team = g.away_team_id THEN -p.pull_x ELSE p.pull_x END AS norm_pull_x,
                    CASE WHEN recv.team = g.away_team_id THEN 120 - p.pull_y ELSE p.pull_y END AS norm_pull_y,
                    recv.team AS recv_team,
                    CASE WHEN recv.team = g.away_team_id THEN TRUE ELSE FALSE END AS is_away
                FROM events p
                JOIN games g ON p.game_id = g.game_id
                JOIN events recv ON p.game_id = recv.game_id
                    AND recv.event_number = p.event_number + 1
                    AND recv.event_type IN (18, 19, 20, 22)
                WHERE p.event_type = 7
                  AND p.pull_x IS NOT NULL AND p.pull_y IS NOT NULL
            ),
            nearby_pulls AS (
                SELECT *
                FROM pull_data
                WHERE SQRT(POWER(norm_pull_x - %s, 2) + POWER(norm_pull_y - %s, 2)) <= %s
                {team_filter}
            ),
            throw_sequences AS (
                SELECT
                    np.game_id, np.pull_num, np.is_away,
                    CASE WHEN np.is_away THEN -e1.thrower_x ELSE e1.thrower_x END AS t1_fx,
                    CASE WHEN np.is_away THEN 120 - e1.thrower_y ELSE e1.thrower_y END AS t1_fy,
                    CASE WHEN np.is_away THEN -e1.receiver_x ELSE e1.receiver_x END AS t1_tx,
                    CASE WHEN np.is_away THEN 120 - e1.receiver_y ELSE e1.receiver_y END AS t1_ty,
                    CASE WHEN np.is_away THEN -e2.thrower_x ELSE e2.thrower_x END AS t2_fx,
                    CASE WHEN np.is_away THEN 120 - e2.thrower_y ELSE e2.thrower_y END AS t2_fy,
                    CASE WHEN np.is_away THEN -e2.receiver_x ELSE e2.receiver_x END AS t2_tx,
                    CASE WHEN np.is_away THEN 120 - e2.receiver_y ELSE e2.receiver_y END AS t2_ty,
                    CASE WHEN np.is_away THEN -e3.thrower_x ELSE e3.thrower_x END AS t3_fx,
                    CASE WHEN np.is_away THEN 120 - e3.thrower_y ELSE e3.thrower_y END AS t3_fy,
                    CASE WHEN np.is_away THEN -e3.receiver_x ELSE e3.receiver_x END AS t3_tx,
                    CASE WHEN np.is_away THEN 120 - e3.receiver_y ELSE e3.receiver_y END AS t3_ty,
                    EXISTS (
                        SELECT 1 FROM events eg
                        WHERE eg.game_id = np.game_id
                          AND eg.event_number > np.pull_num
                          AND eg.event_number <= np.pull_num + 30
                          AND eg.event_type = 19
                          AND eg.team = e1.team
                    ) AS scored
                FROM nearby_pulls np
                JOIN events e1 ON np.game_id = e1.game_id AND e1.event_number = np.pull_num + 1
                JOIN events e2 ON np.game_id = e2.game_id AND e2.event_number = np.pull_num + 2
                JOIN events e3 ON np.game_id = e3.game_id AND e3.event_number = np.pull_num + 3
                WHERE e1.event_type IN (18, 19) AND e1.receiver_x IS NOT NULL
                  AND e2.event_type IN (18, 19) AND e2.receiver_x IS NOT NULL
                  AND e3.event_type IN (18, 19) AND e3.receiver_x IS NOT NULL
                  AND e1.team = e2.team AND e2.team = e3.team
            )
            SELECT
                AVG(t1_fx) AS t1_fx, AVG(t1_fy) AS t1_fy, AVG(t1_tx) AS t1_tx, AVG(t1_ty) AS t1_ty,
                AVG(t2_fx) AS t2_fx, AVG(t2_fy) AS t2_fy, AVG(t2_tx) AS t2_tx, AVG(t2_ty) AS t2_ty,
                AVG(t3_fx) AS t3_fx, AVG(t3_fy) AS t3_fy, AVG(t3_tx) AS t3_tx, AVG(t3_ty) AS t3_ty,
                COUNT(*) AS sample_size,
                AVG(CASE WHEN scored THEN 1.0 ELSE 0.0 END) AS scoring_rate
            FROM throw_sequences
        """, params)

        row = cur.fetchone()
        cur.close()
        conn.close()

        if not row or row['sample_size'] == 0:
            return {
                "throws": [],
                "sample_size": 0,
                "pull_landing": {"x": pull_x, "y": pull_y},
                "scoring_rate": 0,
            }

        return {
            "throws": [
                {"from_x": round(float(row['t1_fx']), 1), "from_y": round(float(row['t1_fy']), 1),
                 "to_x": round(float(row['t1_tx']), 1), "to_y": round(float(row['t1_ty']), 1)},
                {"from_x": round(float(row['t2_fx']), 1), "from_y": round(float(row['t2_fy']), 1),
                 "to_x": round(float(row['t2_tx']), 1), "to_y": round(float(row['t2_ty']), 1)},
                {"from_x": round(float(row['t3_fx']), 1), "from_y": round(float(row['t3_fy']), 1),
                 "to_x": round(float(row['t3_tx']), 1), "to_y": round(float(row['t3_ty']), 1)},
            ],
            "sample_size": int(row['sample_size']),
            "pull_landing": {"x": pull_x, "y": pull_y},
            "scoring_rate": round(float(row['scoring_rate']), 3),
            "mode": "average",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/pull-play/sample")
def sample_pull_plays(
    pull_x: float = Query(0.0, description="Pull landing X position (-25 to 25)"),
    pull_y: float = Query(20.0, description="Pull landing Y position (0 to 120)"),
    team: Optional[str] = Query(None, description="Receiving team (omit for all teams)"),
    n_samples: int = Query(5, ge=1, le=20, description="Number of sequences to sample"),
) -> Dict[str, Any]:
    """Sample multiple distinct play sequences from the CVAE latent space."""
    try:
        sequences = _cvae_predict(pull_x, pull_y, team, n_samples=n_samples)
        return {
            "sequences": sequences,
            "pull_landing": {"x": pull_x, "y": pull_y},
            "n_samples": n_samples,
            "team": team,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/possession/zone-patterns")
def get_zone_patterns(
    team: Optional[str] = Query(None, description="Receiving team (omit for all teams)"),
    zone_cols: int = Query(4, ge=2, le=6),
    zone_rows: int = Query(3, ge=2, le=5),
) -> Dict[str, Any]:
    """
    Divide the playing field into a grid of zones. For each zone, compute the
    average 3-throw sequence from all possession starts (pulls + turnovers)
    that originate in that zone. Direction-normalized, offense attacks y=120.
    """
    tf = "AND off_team = %s" if team else ""
    params: list = [team] if team else []

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"""
        WITH poss_starts AS (
            -- Possession starts after pulls: receiving team's first throw
            SELECT
                e2.game_id,
                e2.event_number AS e1_num,
                e2.team AS off_team,
                CASE WHEN e2.team = g.away_team_id THEN TRUE ELSE FALSE END AS is_away
            FROM events p
            JOIN games g ON p.game_id = g.game_id
            JOIN events e2 ON p.game_id = e2.game_id
                AND e2.event_number = p.event_number + 1
                AND e2.event_type IN (18, 19, 20, 22)
            WHERE p.event_type = 7

            UNION ALL

            -- Possession starts after turnovers: new possessing team's first throw
            SELECT
                e2.game_id,
                e2.event_number AS e1_num,
                e2.team AS off_team,
                CASE WHEN e2.team = g.away_team_id THEN TRUE ELSE FALSE END AS is_away
            FROM events t
            JOIN games g ON t.game_id = g.game_id
            JOIN events e2 ON t.game_id = e2.game_id
                AND e2.event_number = t.event_number + 1
                AND e2.team != t.team
                AND e2.event_type IN (18, 19, 20, 22)
            WHERE t.event_type IN (20, 22, 11)
        ),
        seqs AS (
            SELECT
                ps.off_team,
                CASE WHEN ps.is_away THEN -e1.thrower_x  ELSE e1.thrower_x  END AS t1_fx,
                CASE WHEN ps.is_away THEN 120-e1.thrower_y ELSE e1.thrower_y END AS t1_fy,
                CASE WHEN ps.is_away THEN -e1.receiver_x ELSE e1.receiver_x END AS t1_tx,
                CASE WHEN ps.is_away THEN 120-e1.receiver_y ELSE e1.receiver_y END AS t1_ty,
                CASE WHEN ps.is_away THEN -e2.thrower_x  ELSE e2.thrower_x  END AS t2_fx,
                CASE WHEN ps.is_away THEN 120-e2.thrower_y ELSE e2.thrower_y END AS t2_fy,
                CASE WHEN ps.is_away THEN -e2.receiver_x ELSE e2.receiver_x END AS t2_tx,
                CASE WHEN ps.is_away THEN 120-e2.receiver_y ELSE e2.receiver_y END AS t2_ty,
                CASE WHEN ps.is_away THEN -e3.thrower_x  ELSE e3.thrower_x  END AS t3_fx,
                CASE WHEN ps.is_away THEN 120-e3.thrower_y ELSE e3.thrower_y END AS t3_fy,
                CASE WHEN ps.is_away THEN -e3.receiver_x ELSE e3.receiver_x END AS t3_tx,
                CASE WHEN ps.is_away THEN 120-e3.receiver_y ELSE e3.receiver_y END AS t3_ty
            FROM poss_starts ps
            JOIN events e1 ON ps.game_id = e1.game_id AND e1.event_number = ps.e1_num
            JOIN events e2 ON ps.game_id = e2.game_id AND e2.event_number = ps.e1_num + 1
            JOIN events e3 ON ps.game_id = e3.game_id AND e3.event_number = ps.e1_num + 2
            WHERE e1.event_type IN (18, 19) AND e1.thrower_x IS NOT NULL AND e1.receiver_x IS NOT NULL
              AND e2.event_type IN (18, 19) AND e2.receiver_x IS NOT NULL
              AND e3.event_type IN (18, 19) AND e3.receiver_x IS NOT NULL
              AND e1.team = e2.team AND e2.team = e3.team
              {tf}
        )
        SELECT * FROM seqs
    """, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="No sequence data found")

    SEQ_COLS = ['t1_fx','t1_fy','t1_tx','t1_ty','t2_fx','t2_fy','t2_tx','t2_ty','t3_fx','t3_fy','t3_tx','t3_ty']
    X = np.array([[float(r[c]) for c in SEQ_COLS] for r in rows], dtype=np.float32)

    # Playing field: x=-25..25, y=10..110 (exclude end zones for zone assignment)
    x_edges = np.linspace(-25, 25, zone_cols + 1)
    y_edges = np.linspace(10, 110, zone_rows + 1)

    zones = []
    for row_i in range(zone_rows):
        for col_i in range(zone_cols):
            x_lo, x_hi = float(x_edges[col_i]), float(x_edges[col_i + 1])
            y_lo, y_hi = float(y_edges[row_i]), float(y_edges[row_i + 1])
            mask = (
                (X[:, 0] >= x_lo) & (X[:, 0] < x_hi) &
                (X[:, 1] >= y_lo) & (X[:, 1] < y_hi)
            )
            count = int(mask.sum())
            if count < 5:
                zones.append({
                    "zone_id": row_i * zone_cols + col_i,
                    "col": col_i, "row": row_i,
                    "x_range": [x_lo, x_hi], "y_range": [y_lo, y_hi],
                    "count": 0, "throws": []
                })
                continue
            center = X[mask].mean(axis=0)
            throws = []
            for t in range(3):
                throws.append({
                    "from_x": round(float(center[t*4+0]), 1),
                    "from_y": round(float(center[t*4+1]), 1),
                    "to_x":   round(float(center[t*4+2]), 1),
                    "to_y":   round(float(center[t*4+3]), 1),
                })
            zones.append({
                "zone_id": row_i * zone_cols + col_i,
                "col": col_i, "row": row_i,
                "x_range": [x_lo, x_hi], "y_range": [y_lo, y_hi],
                "count": count, "throws": throws,
            })

    max_count = max((z["count"] for z in zones), default=1)
    for z in zones:
        z["relative_density"] = round(z["count"] / max_count, 3)

    return {"zones": zones, "zone_cols": zone_cols, "zone_rows": zone_rows, "total": len(rows)}


@app.get("/pull-play/hotspots")
def get_pull_play_hotspots(
    team: Optional[str] = Query(None, description="Receiving team (omit for all teams)"),
    top_n: int = Query(20, ge=5, le=50),
) -> Dict[str, Any]:
    """
    Return the most common pull landing positions (direction-normalized).
    These are the hotspots where teams most frequently run startup plays.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    tf = "AND recv.team = %s" if team else ""
    params: list = [team] if team else []
    cur.execute(f"""
        SELECT
            CASE WHEN recv.team = g.away_team_id THEN -p.pull_x ELSE p.pull_x END AS norm_x,
            CASE WHEN recv.team = g.away_team_id THEN 120 - p.pull_y ELSE p.pull_y END AS norm_y,
            COUNT(*) AS cnt
        FROM events p
        JOIN games g ON p.game_id = g.game_id
        JOIN events recv ON p.game_id = recv.game_id
            AND recv.event_number = p.event_number + 1
            AND recv.event_type IN (18, 19, 20, 22)
        WHERE p.event_type = 7
          AND p.pull_x IS NOT NULL AND p.pull_y IS NOT NULL
          {tf}
        GROUP BY norm_x, norm_y
        ORDER BY cnt DESC
        LIMIT %s
    """, params + [top_n * 5])  # fetch more, then bin
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return {"hotspots": []}

    # Bin nearby positions together (within 5 yards) using greedy merging
    hotspots: list = []
    for row in rows:
        rx, ry, cnt = float(row['norm_x']), float(row['norm_y']), int(row['cnt'])
        merged = False
        for h in hotspots:
            if (rx - h['x']) ** 2 + (ry - h['y']) ** 2 < 25:  # within 5 yards
                # Weighted average position
                total = h['count'] + cnt
                h['x'] = (h['x'] * h['count'] + rx * cnt) / total
                h['y'] = (h['y'] * h['count'] + ry * cnt) / total
                h['count'] = total
                merged = True
                break
        if not merged:
            hotspots.append({'x': rx, 'y': ry, 'count': cnt})

    hotspots.sort(key=lambda h: -h['count'])
    hotspots = hotspots[:top_n]
    max_count = hotspots[0]['count'] if hotspots else 1
    for h in hotspots:
        h['x'] = round(h['x'], 1)
        h['y'] = round(h['y'], 1)
        h['relative_freq'] = round(h['count'] / max_count, 3)

    return {"hotspots": hotspots}


@app.get("/pull-play/clusters")
def get_pull_play_clusters(
    pull_x: float = Query(0.0, description="Pull landing X (-25 to 25)"),
    pull_y: float = Query(20.0, description="Pull landing Y (0 to 120)"),
    team: Optional[str] = Query(None, description="Receiving team (omit for all teams)"),
) -> Dict[str, Any]:
    """
    Cluster actual 3-throw sequences from near this pull landing (or for this team
    league-wide when too few sequences exist near the position).
    Uses HDBSCAN to automatically detect the natural number of play archetypes.
    """
    from sklearn.cluster import KMeans as _KMeans
    from sklearn.metrics import silhouette_score as _silhouette_score
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    SEQ_COLS = ['t1_fx','t1_fy','t1_tx','t1_ty',
                't2_fx','t2_fy','t2_tx','t2_ty',
                't3_fx','t3_fy','t3_tx','t3_ty']

    def _query_sequences(radius: float, team_filter: Optional[str]) -> list:
        conn = get_db_connection()
        cur = conn.cursor()
        tf = "AND recv_team = %s" if team_filter else ""
        params: list = [pull_x, pull_y, radius]
        if team_filter:
            params.append(team_filter)
        cur.execute(f"""
            WITH pull_data AS (
                SELECT
                    p.game_id, p.event_number AS pull_num,
                    CASE WHEN recv.team = g.away_team_id THEN -p.pull_x ELSE p.pull_x END AS norm_pull_x,
                    CASE WHEN recv.team = g.away_team_id THEN 120 - p.pull_y ELSE p.pull_y END AS norm_pull_y,
                    recv.team AS recv_team,
                    CASE WHEN recv.team = g.away_team_id THEN TRUE ELSE FALSE END AS is_away
                FROM events p
                JOIN games g ON p.game_id = g.game_id
                JOIN events recv ON p.game_id = recv.game_id
                    AND recv.event_number = p.event_number + 1
                    AND recv.event_type IN (18, 19, 20, 22)
                WHERE p.event_type = 7
                  AND p.pull_x IS NOT NULL AND p.pull_y IS NOT NULL
            ),
            nearby AS (
                SELECT * FROM pull_data
                WHERE SQRT(POWER(norm_pull_x - %s, 2) + POWER(norm_pull_y - %s, 2)) <= %s
                {tf}
            ),
            seqs AS (
                SELECT
                    CASE WHEN n.is_away THEN -e1.thrower_x  ELSE e1.thrower_x  END AS t1_fx,
                    CASE WHEN n.is_away THEN 120-e1.thrower_y ELSE e1.thrower_y END AS t1_fy,
                    CASE WHEN n.is_away THEN -e1.receiver_x ELSE e1.receiver_x END AS t1_tx,
                    CASE WHEN n.is_away THEN 120-e1.receiver_y ELSE e1.receiver_y END AS t1_ty,
                    CASE WHEN n.is_away THEN -e2.thrower_x  ELSE e2.thrower_x  END AS t2_fx,
                    CASE WHEN n.is_away THEN 120-e2.thrower_y ELSE e2.thrower_y END AS t2_fy,
                    CASE WHEN n.is_away THEN -e2.receiver_x ELSE e2.receiver_x END AS t2_tx,
                    CASE WHEN n.is_away THEN 120-e2.receiver_y ELSE e2.receiver_y END AS t2_ty,
                    CASE WHEN n.is_away THEN -e3.thrower_x  ELSE e3.thrower_x  END AS t3_fx,
                    CASE WHEN n.is_away THEN 120-e3.thrower_y ELSE e3.thrower_y END AS t3_fy,
                    CASE WHEN n.is_away THEN -e3.receiver_x ELSE e3.receiver_x END AS t3_tx,
                    CASE WHEN n.is_away THEN 120-e3.receiver_y ELSE e3.receiver_y END AS t3_ty
                FROM nearby n
                JOIN events e1 ON n.game_id = e1.game_id AND e1.event_number = n.pull_num + 1
                JOIN events e2 ON n.game_id = e2.game_id AND e2.event_number = n.pull_num + 2
                JOIN events e3 ON n.game_id = e3.game_id AND e3.event_number = n.pull_num + 3
                WHERE e1.event_type IN (18,19) AND e1.receiver_x IS NOT NULL
                  AND e2.event_type IN (18,19) AND e2.receiver_x IS NOT NULL
                  AND e3.event_type IN (18,19) AND e3.receiver_x IS NOT NULL
                  AND e1.team = e2.team AND e2.team = e3.team
            )
            SELECT * FROM seqs
        """, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows

    # For team queries: first try a radius of 30; if < 50 sequences expand to whole field
    # For all-teams: use radius 25 (position-specific)
    if team:
        rows = _query_sequences(30.0, team)
        if len(rows) < 50:
            rows = _query_sequences(200.0, team)  # whole field for this team
    else:
        rows = _query_sequences(25.0, None)

    if len(rows) < 10:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough sequences ({len(rows)}) to cluster. Try a different pull position."
        )

    X = np.array([[float(r[c]) for c in SEQ_COLS] for r in rows], dtype=np.float32)
    total = len(rows)

    # Normalize so x and y coords contribute equally (y range is 5x wider than x)
    scaler = _StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try k=2..6, pick the k with the highest silhouette score.
    # Use a subsample for silhouette scoring to keep it fast on large datasets.
    max_k = min(6, total // 5)  # need at least 5 points per cluster
    if max_k < 2:
        raise HTTPException(
            status_code=404,
            detail=f"Not enough sequences ({total}) to cluster. Try a different pull position."
        )

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(total, size=min(500, total), replace=False)
    X_sample = X_scaled[sample_idx]

    best_k, best_score, best_labels = 2, -1.0, None
    for k in range(2, max_k + 1):
        km = _KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(X_scaled)
        score = float(_silhouette_score(X_sample, lbls[sample_idx]))
        if score > best_score:
            best_k, best_score, best_labels = k, score, lbls

    counts = np.bincount(best_labels, minlength=best_k)
    order = np.argsort(-counts)

    result_clusters = []
    for rank, i in enumerate(order):
        mask = best_labels == i
        center = X[mask].mean(axis=0)  # original coords for output
        throws = []
        for t in range(3):
            throws.append({
                "from_x": round(float(center[t*4+0]), 1),
                "from_y": round(float(center[t*4+1]), 1),
                "to_x":   round(float(center[t*4+2]), 1),
                "to_y":   round(float(center[t*4+3]), 1),
            })
        result_clusters.append({
            "cluster_id": rank + 1,
            "throws": throws,
            "count": int(counts[i]),
            "frequency": round(float(counts[i]) / total, 3),
        })

    return {"clusters": result_clusters, "n_clusters": best_k, "sample_size": total}


def _epv_predict_batch(grid_points: np.ndarray, model_type: str) -> np.ndarray:
    """Run EPV inference on a batch of feature rows."""
    if model_type == "xgb":
        return _epv_xgb.predict_proba(grid_points)[:, 1]
    else:
        scaled = _epv_scaler.transform(grid_points)
        _epv_nn.eval()
        with torch.no_grad():
            return _epv_nn(torch.FloatTensor(scaled)).numpy()


@app.get("/epv/heatmap")
def get_epv_heatmap(
    throw_idx: int = Query(1, ge=1, le=10, description="Throw number within possession (1-10)"),
    team: Optional[str] = Query(None, description="Team name for team-specific EPV (omit for league average)"),
    model: str = Query("xgb", description="Model to use: 'xgb' or 'nn'"),
    quarter: Optional[int] = Query(None, ge=1, le=4, description="Game quarter 1-4 (omit for average over all quarters)"),
) -> Dict[str, Any]:
    """
    Return EPV probability grid over the field.
    Grid shape: 60 rows (y) × 25 cols (x), values in [0, 1].
    """
    if model == "xgb" and _epv_xgb is None:
        raise HTTPException(status_code=503, detail="EPV XGBoost model not loaded. Run epv_model.ipynb first.")
    if model == "nn" and _epv_nn is None:
        raise HTTPException(status_code=503, detail="EPV Neural Net model not loaded. Run epv_model.ipynb first.")
    if _epv_team_encoder is None:
        raise HTTPException(status_code=503, detail="EPV team encoder not loaded.")

    try:
        xs = np.linspace(-25, 25, 25)
        ys = np.linspace(0, 120, 60)
        xx, yy = np.meshgrid(xs, ys)  # both (60, 25)
        n_points = xx.size  # 1500

        quarters_to_avg = [quarter] if quarter is not None else [1, 2, 3, 4]
        teams_to_avg: list

        if team is not None:
            try:
                team_id = int(_epv_team_encoder.transform([team])[0])
            except (ValueError, KeyError):
                raise HTTPException(status_code=400, detail=f"Unknown team: {team}")
            teams_to_avg = [team_id]
        else:
            teams_to_avg = list(range(len(_epv_team_encoder.classes_)))

        combos = len(quarters_to_avg) * len(teams_to_avg)
        all_probs = np.zeros((combos, n_points), dtype=np.float32)
        idx = 0
        for q in quarters_to_avg:
            for tid in teams_to_avg:
                gp = np.column_stack([
                    xx.ravel(), yy.ravel(),
                    np.full(n_points, throw_idx),
                    np.zeros(n_points),   # prev_throw_dx (neutral)
                    np.zeros(n_points),   # prev_throw_dy (neutral)
                    np.full(n_points, tid),
                    np.full(n_points, q),
                ]).astype(np.float32)
                all_probs[idx] = _epv_predict_batch(gp, model)
                idx += 1

        grid = all_probs.mean(axis=0).reshape(60, 25)

        return {
            "grid": [[round(float(v), 4) for v in row] for row in grid],
            "extent": [-25.0, 25.0, 0.0, 120.0],
            "throw_idx": throw_idx,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Completion % Endpoints
# ---------------------------------------------------------------------------

@app.get("/completion/throwers")
def get_completion_throwers() -> List[str]:
    """Return sorted list of thrower names known to the completion model."""
    if _completion_thrower_encoder is None:
        raise HTTPException(status_code=503, detail="Completion model not loaded. Run completion_model.ipynb first.")
    return sorted(_completion_thrower_encoder.classes_.tolist())


@app.get("/completion/heatmap")
def get_completion_heatmap(
    thrower: str = Query(..., description="Thrower name"),
    from_x: float = Query(..., description="Throw origin X (-25 to 25)"),
    from_y: float = Query(..., description="Throw origin Y (0 to 120)"),
) -> Dict[str, Any]:
    """
    Return completion probability grid for all target positions from a given origin.
    Grid shape: 60 rows (y) × 25 cols (x), values in [0, 1].
    """
    if _completion_xgb is None or _completion_thrower_encoder is None:
        raise HTTPException(status_code=503, detail="Completion model not loaded. Run completion_model.ipynb first.")

    try:
        if thrower in _completion_thrower_encoder.classes_:
            thrower_enc = int(_completion_thrower_encoder.transform([thrower])[0])
        else:
            thrower_enc = 0  # fallback to first encoded class

        xs = np.linspace(-25, 25, 25)
        ys = np.linspace(0, 120, 60)
        xx, yy = np.meshgrid(xs, ys)  # (60, 25)

        to_x = xx.ravel()
        to_y = yy.ravel()
        dy = to_y - from_y
        dx = to_x - from_x
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)

        grid_points = np.column_stack([
            np.full(xx.size, from_x),
            np.full(xx.size, from_y),
            to_x, to_y,
            dist, dy, dx, angle,
            np.full(xx.size, thrower_enc),
        ]).astype(np.float32)

        probs = _completion_xgb.predict_proba(grid_points)[:, 1]
        grid = probs.reshape(60, 25)

        return {
            "grid": [[round(float(v), 4) for v in row] for row in grid],
            "extent": [-25.0, 25.0, 0.0, 120.0],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/completion/predict")
def get_completion_predict(
    thrower: str = Query(..., description="Thrower name"),
    from_x: float = Query(..., description="Throw origin X (-25 to 25)"),
    from_y: float = Query(..., description="Throw origin Y (0 to 120)"),
    to_x: float = Query(..., description="Throw target X (-25 to 25)"),
    to_y: float = Query(..., description="Throw target Y (0 to 120)"),
) -> Dict[str, Any]:
    """Return completion probability for a single from→to throw by a specific thrower."""
    if _completion_xgb is None or _completion_thrower_encoder is None:
        raise HTTPException(status_code=503, detail="Completion model not loaded. Run completion_model.ipynb first.")

    try:
        if thrower in _completion_thrower_encoder.classes_:
            thrower_enc = int(_completion_thrower_encoder.transform([thrower])[0])
        else:
            thrower_enc = 0

        dy = to_y - from_y
        dx = to_x - from_x
        dist = float(np.sqrt(dx**2 + dy**2))
        angle = float(np.arctan2(dy, dx))

        row = np.array([[from_x, from_y, to_x, to_y, dist, dy, dx, angle, thrower_enc]], dtype=np.float32)
        prob = float(_completion_xgb.predict_proba(row)[0, 1])

        return {"probability": round(prob, 4), "thrower": thrower}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
