from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import psycopg2
import base64
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import numpy as np
import joblib
from pathlib import Path
from scipy.ndimage import gaussian_filter


# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="UFA Analytics API")

# CORS - allow frontend origins (local dev + production)
_cors_origins = ["http://localhost:5173"]
for _url in (os.getenv("FRONTEND_URL", ""), os.getenv("FRONTEND_URL_WWW", "")):
    if _url:
        _cors_origins.append(_url)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection — Railway provides DATABASE_URL; fall back to individual vars for local dev
_database_url = os.getenv("DATABASE_URL")

if _database_url:
    # Railway format: postgresql://user:pass@host:port/dbname
    import urllib.parse
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

def get_db_connection():
    return psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)


# ---------------------------------------------------------------------------
# ML Model Loading
# ---------------------------------------------------------------------------

# Throw flow model — lazy-loaded on first cache miss
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "normalizing_flow_model.pkl"

_flow = None
_context_net = None
_player_encoder = None
_player_names: List[str] = []
_umap_coordinates: np.ndarray | None = None
_cluster_labels: np.ndarray | None = None
_flow_load_attempted = False

def _ensure_flow_loaded():
    """Load the throw flow model on first use (lazy). Returns True if model is available."""
    global _flow, _context_net, _player_encoder, _player_names, _flow_load_attempted
    if _flow is not None:
        return True
    if _flow_load_attempted:
        return False
    _flow_load_attempted = True
    try:
        import torch
        import torch.nn as nn
        from nflows import flows, distributions, transforms

        class ContextNetwork(nn.Module):
            def __init__(self, n_players, embedding_dim=16, hidden_dim=64, output_dim=32):
                super().__init__()
                self.player_embedding = nn.Embedding(n_players, embedding_dim)
                self.network = nn.Sequential(
                    nn.Linear(embedding_dim + 2, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
            def forward(self, context):
                player_emb = self.player_embedding(context[:, 0].long())
                return self.network(torch.cat([player_emb, context[:, 1:3]], dim=1))

        save_dict = joblib.load(MODEL_PATH)
        hp = save_dict["hyperparameters"]
        _player_encoder = save_dict["player_encoder"]

        base_dist = distributions.StandardNormal(shape=[2])
        ctx_net = ContextNetwork(n_players=hp["n_players"], embedding_dim=16, hidden_dim=64, output_dim=hp["context_features"])
        tlist = []
        for _ in range(hp["num_layers"]):
            tlist.append(transforms.MaskedAffineAutoregressiveTransform(
                features=2, hidden_features=hp["hidden_features"], context_features=hp["context_features"], num_blocks=2))
            tlist.append(transforms.ReversePermutation(features=2))
        flow = flows.Flow(transforms.CompositeTransform(tlist), base_dist)

        flow.load_state_dict(save_dict["flow_state_dict"])
        ctx_net.load_state_dict(save_dict["context_net_state_dict"])
        flow.eval()
        ctx_net.eval()
        _flow = flow
        _context_net = ctx_net
        _player_names = sorted(_player_encoder.classes_.tolist())
        print(f"[lazy] Throw flow model loaded from {MODEL_PATH} ({len(_player_names)} players)")
        return True
    except Exception as e:
        print(f"Warning: Could not load throw flow model: {e}")
        return False

# Load player encoder only (lightweight — needed for /players endpoint and UMAP)
try:
    _encoder_save = joblib.load(MODEL_PATH)
    _player_encoder = _encoder_save["player_encoder"]
    _player_names = sorted(_player_encoder.classes_.tolist())
    print(f"Player encoder loaded ({len(_player_names)} players)")
except Exception as e:
    print(f"Warning: Could not load player encoder: {e}")

# Turnover flow model — lazy-loaded on first cache miss
TURNOVER_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "turnover_flow_model.pkl"

_turnover_flow = None
_turnover_context_net = None
_turnover_load_attempted = False

def _ensure_turnover_flow_loaded():
    global _turnover_flow, _turnover_context_net, _turnover_load_attempted
    if _turnover_flow is not None:
        return True
    if _turnover_load_attempted:
        return False
    _turnover_load_attempted = True
    try:
        import torch.nn as nn
        from nflows import flows, distributions, transforms

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

        turnover_save = joblib.load(TURNOVER_MODEL_PATH)
        hp = turnover_save["hyperparameters"]
        ctx_net = TurnoverContextNetwork(hidden_dim=64, output_dim=hp["context_features"])
        tlist = []
        for _ in range(hp["num_layers"]):
            tlist.append(transforms.MaskedAffineAutoregressiveTransform(
                features=2, hidden_features=hp["hidden_features"], context_features=hp["context_features"], num_blocks=2))
            tlist.append(transforms.ReversePermutation(features=2))
        flow = flows.Flow(transforms.CompositeTransform(tlist), distributions.StandardNormal(shape=[2]))
        flow.load_state_dict(turnover_save["flow_state_dict"])
        ctx_net.load_state_dict(turnover_save["context_net_state_dict"])
        flow.eval()
        ctx_net.eval()
        _turnover_flow = flow
        _turnover_context_net = ctx_net
        print(f"[lazy] Turnover flow model loaded from {TURNOVER_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Warning: Could not load turnover flow model: {e}")
        return False

# Block flow model — lazy-loaded on first cache miss
BLOCK_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "block_flow_model.pkl"

_block_flow = None
_block_context_net = None
_block_load_attempted = False

def _ensure_block_flow_loaded():
    global _block_flow, _block_context_net, _block_load_attempted
    if _block_flow is not None:
        return True
    if _block_load_attempted:
        return False
    _block_load_attempted = True
    try:
        import torch.nn as nn
        from nflows import flows, distributions, transforms

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

        block_save = joblib.load(BLOCK_MODEL_PATH)
        hp = block_save["hyperparameters"]
        ctx_net = TurnoverContextNetwork(hidden_dim=64, output_dim=hp["context_features"])
        tlist = []
        for _ in range(hp["num_layers"]):
            tlist.append(transforms.MaskedAffineAutoregressiveTransform(
                features=2, hidden_features=hp["hidden_features"], context_features=hp["context_features"], num_blocks=2))
            tlist.append(transforms.ReversePermutation(features=2))
        flow = flows.Flow(transforms.CompositeTransform(tlist), distributions.StandardNormal(shape=[2]))
        flow.load_state_dict(block_save["flow_state_dict"])
        ctx_net.load_state_dict(block_save["context_net_state_dict"])
        flow.eval()
        ctx_net.eval()
        _block_flow = flow
        _block_context_net = ctx_net
        print(f"[lazy] Block flow model loaded from {BLOCK_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"Warning: Could not load block flow model: {e}")
        return False

# EPV model — lazy-loaded on first cache miss
EPV_NN_PATH = Path(__file__).resolve().parent.parent / "models" / "epv_nn.pkl"

_epv_nn = None
_epv_scaler = None
_epv_team_encoder = None
_epv_load_attempted = False

def _ensure_epv_loaded():
    global _epv_nn, _epv_scaler, _epv_team_encoder, _epv_load_attempted
    if _epv_nn is not None:
        return True
    if _epv_load_attempted:
        return False
    _epv_load_attempted = True
    try:
        import torch
        import torch.nn as nn

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

        epv_save = joblib.load(EPV_NN_PATH)
        _epv_scaler = epv_save["scaler"]
        _epv_team_encoder = epv_save["team_encoder"]
        _epv_nn = EPVNet(input_dim=epv_save.get("input_dim", 6))
        _epv_nn.load_state_dict(epv_save["model_state_dict"])
        _epv_nn.eval()
        print(f"[lazy] EPV Neural Net loaded (AUC={epv_save['metrics']['auc']:.4f})")
        return True
    except Exception as e:
        print(f"Warning: Could not load EPV neural net model: {e}")
        return False

# Completion NN model definition (must match completion_nn.ipynb)
# Load lineup XGBoost model (P(score) for a 7-player O-lineup)
LINEUP_XGB_PATH = Path(__file__).resolve().parent.parent / "models" / "lineup_xgb.pkl"

_lineup_xgb = None
_lineup_player_stats: Dict[str, Dict[str, float]] = {}
_lineup_stat_cols: List[str] = []
_lineup_feature_names: List[str] = []
_lineup_league_avg: Dict[str, float] = {}
_lineup_pair_familiarity: Dict[tuple, int] = {}

try:
    lineup_xgb_save = joblib.load(LINEUP_XGB_PATH)
    _lineup_xgb = lineup_xgb_save["model"]
    _lineup_player_stats = lineup_xgb_save["player_stats"]
    _lineup_stat_cols = lineup_xgb_save["stat_cols"]
    _lineup_feature_names = lineup_xgb_save["feature_names"]
    _lineup_league_avg = lineup_xgb_save["league_avg"]
    _lineup_pair_familiarity = lineup_xgb_save.get("pair_familiarity", {})
    print(f"Lineup model loaded (AUC={lineup_xgb_save['metrics']['auc']:.4f}, {len(_lineup_player_stats)} players, {len(_lineup_pair_familiarity)} pairs)")
except Exception as e:
    print(f"Warning: Could not load lineup model: {e}")

# ---------------------------------------------------------------------------
# Player Stats, UMAP, and Clustering (stats-based)
# ---------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler

def _write_umap_cache(players_list: list, coords: np.ndarray, labels: np.ndarray):
    """Write UMAP coordinates and cluster labels to DB, replacing any existing rows."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS umap_cache (
                player_id     VARCHAR PRIMARY KEY,
                umap_x        REAL NOT NULL,
                umap_y        REAL NOT NULL,
                cluster_label SMALLINT NOT NULL
            )
        """)
        cur.execute("DELETE FROM umap_cache")
        from psycopg2.extras import execute_values
        execute_values(cur,
            "INSERT INTO umap_cache (player_id, umap_x, umap_y, cluster_label) VALUES %s",
            [(p, float(coords[i, 0]), float(coords[i, 1]), int(labels[i])) for i, p in enumerate(players_list)]
        )
        conn.commit()
        cur.close()
        conn.close()
        print(f"UMAP cache written: {len(players_list)} players")
    except Exception as e:
        print(f"Warning: Could not write UMAP cache: {e}")

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
                WHEN event_type IN (18, 19) AND receiver_x IS NOT NULL AND receiver_y IS NOT NULL
                    AND SQRT(POWER(receiver_x - thrower_x, 2) + POWER(receiver_y - thrower_y, 2)) >= 40
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

        # 3. Try loading UMAP + cluster labels from DB cache
        cache_conn = get_db_connection()
        cache_cur = cache_conn.cursor()
        cache_cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'umap_cache'
            )
        """)
        table_exists = cache_cur.fetchone()["exists"]

        loaded_from_cache = False
        if table_exists:
            cache_cur.execute("SELECT player_id, umap_x, umap_y, cluster_label FROM umap_cache ORDER BY player_id")
            cache_rows = cache_cur.fetchall()
            if len(cache_rows) == len(players_list):
                cache_map = {r["player_id"]: r for r in cache_rows}
                if all(p in cache_map for p in players_list):
                    _umap_coordinates = np.array([[cache_map[p]["umap_x"], cache_map[p]["umap_y"]] for p in players_list])
                    _cluster_labels = np.array([cache_map[p]["cluster_label"] for p in players_list])
                    loaded_from_cache = True
                    print(f"UMAP loaded from DB cache: {_umap_coordinates.shape}")

        cache_cur.close()
        cache_conn.close()

        if not loaded_from_cache:
            # Compute UMAP + KMeans and write to cache
            import umap as _umap
            scaler = StandardScaler()
            stats_normalized = scaler.fit_transform(stats_array)

            reducer = _umap.UMAP(n_neighbors=15, min_dist=0.01, random_state=42)
            _umap_coordinates = reducer.fit_transform(stats_normalized)
            print(f"UMAP computed on stats features: {_umap_coordinates.shape}")

            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            _cluster_labels = kmeans.fit_predict(stats_normalized)
            print(f"K-Means: 4 clusters, sizes: {[int((_cluster_labels == i).sum()) for i in range(4)]}")

            _write_umap_cache(players_list, _umap_coordinates, _cluster_labels)

        # 4. Compute cluster summaries (fast, derived from already-loaded data)
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


@app.get("/games/{game_id}/roster")
def get_game_roster(game_id: str) -> Dict[str, Any]:
    """Players who appeared for each team in a game."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT
            lp.player_id,
            p.full_name,
            e.team,
            COUNT(*) FILTER (WHERE lp.line_type = 'O') AS o_pts,
            COUNT(*) FILTER (WHERE lp.line_type = 'D') AS d_pts
        FROM line_players lp
        JOIN events e ON e.event_id = lp.event_id
        JOIN players p ON p.player_id = lp.player_id
        WHERE e.game_id = %s
        GROUP BY lp.player_id, p.full_name, e.team
        ORDER BY e.team, (COUNT(*) FILTER (WHERE lp.line_type = 'O') + COUNT(*) FILTER (WHERE lp.line_type = 'D')) DESC
    """, (game_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    by_team: Dict[str, list] = {}
    for r in rows:
        t = r["team"]
        if t not in by_team:
            by_team[t] = []
        by_team[t].append({
            "id": r["player_id"],
            "name": r["full_name"],
            "o_pts": int(r["o_pts"] or 0),
            "d_pts": int(r["d_pts"] or 0),
        })
    return by_team


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
def get_players():
    """Get list of players available for throw prediction, with full names."""
    if not _player_names:
        raise HTTPException(status_code=503, detail="Model not loaded")
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT player_id, full_name FROM players WHERE player_id = ANY(%s)", (_player_names,))
    name_map = {r["player_id"]: r["full_name"] for r in cur.fetchall()}
    cur.close()
    conn.close()
    return [{"id": p, "name": name_map.get(p, p)} for p in _player_names]


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


@app.get("/debug/player-possessions")
def debug_player_possessions(player_id: str, team: str):
    """Debug: count O-line possessions per year for a player on a team."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT e.year,
               COUNT(DISTINCT lp.event_id) AS distinct_event_ids,
               COUNT(*)                   AS raw_rows
        FROM line_players lp
        JOIN events e ON e.event_id = lp.event_id
        WHERE lp.player_id = %s
          AND lp.line_type = 'O'
          AND e.team = %s
          AND e.event_type = 2
        GROUP BY e.year
        ORDER BY e.year
    """, (player_id, team))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"year": r['year'], "distinct_event_ids": r['distinct_event_ids'], "raw_rows": r['raw_rows']} for r in rows]


@app.get("/team/{team_id}")
def get_team(team_id: str, year: Optional[int] = None):
    """Team overview: record, O-line rate, top players by hold rate, top 5 pair synergies."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Team info
    cur.execute("SELECT team_id, team_name, division FROM teams WHERE team_id = %s", (team_id,))
    team_row = cur.fetchone()
    if not team_row:
        raise HTTPException(status_code=404, detail=f"Team '{team_id}' not found")

    # Available seasons for this team
    cur.execute("""
        SELECT ARRAY_AGG(DISTINCT e.year ORDER BY e.year) AS years
        FROM events e WHERE e.team = %s AND e.year IS NOT NULL
    """, (team_id,))
    years_row = cur.fetchone()
    available_years = years_row['years'] or []

    year_filter = "AND e.year = %s" if year else ""
    year_params_1 = (team_id, year) if year else (team_id,)

    # Win/loss record
    year_game_filter = "AND year = %s" if year else ""
    record_params = (team_id, team_id, team_id, team_id, team_id, team_id)
    if year:
        cur.execute(f"""
            SELECT
                COUNT(*) FILTER (
                    WHERE (home_team_id = %s AND home_score > away_score)
                       OR (away_team_id = %s AND away_score > home_score)
                ) AS wins,
                COUNT(*) FILTER (
                    WHERE (home_team_id = %s AND home_score < away_score)
                       OR (away_team_id = %s AND away_score < home_score)
                ) AS losses,
                COUNT(*) AS total_games
            FROM games
            WHERE (home_team_id = %s OR away_team_id = %s)
              AND home_score IS NOT NULL AND away_score IS NOT NULL
              {year_game_filter}
        """, record_params + ((year,) if year else ()))
    else:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (
                    WHERE (home_team_id = %s AND home_score > away_score)
                       OR (away_team_id = %s AND away_score > home_score)
                ) AS wins,
                COUNT(*) FILTER (
                    WHERE (home_team_id = %s AND home_score < away_score)
                       OR (away_team_id = %s AND away_score < home_score)
                ) AS losses,
                COUNT(*) AS total_games
            FROM games
            WHERE (home_team_id = %s OR away_team_id = %s)
              AND home_score IS NOT NULL AND away_score IS NOT NULL
        """, record_params)
    record = cur.fetchone()

    # O-line scoring rate (direct from team possessions, no player threshold)
    cur.execute(f"""
        WITH o_starts AS (
            SELECT e.event_id, e.game_id, e.event_number, e.team
            FROM events e WHERE e.event_type = 2 AND e.team = %s {year_filter}
        ),
        outcomes AS (
            SELECT os.event_id,
                COALESCE(BOOL_OR(e2.event_type = 19 AND e2.team = os.team), FALSE) AS scored
            FROM o_starts os
            LEFT JOIN events e2 ON e2.game_id = os.game_id
                AND e2.event_number > os.event_number
                AND e2.event_number < COALESCE(
                    (SELECT MIN(e3.event_number) FROM events e3
                     WHERE e3.game_id = os.game_id AND e3.event_type = 1
                       AND e3.event_number > os.event_number), 9999999
                )
            GROUP BY os.event_id
        )
        SELECT ROUND(AVG(CASE WHEN scored THEN 1.0 ELSE 0.0 END)::numeric, 4) AS hold_rate
        FROM outcomes
    """, year_params_1)
    o_line_rate = float(cur.fetchone()['hold_rate'] or 0)

    # Top players by hold rate
    cur.execute(f"""
        WITH o_starts AS (
            SELECT e.event_id, e.game_id, e.event_number, e.team
            FROM events e WHERE e.event_type = 2 AND e.team = %s {year_filter}
        ),
        outcomes AS (
            SELECT os.event_id,
                COALESCE(BOOL_OR(e2.event_type = 19 AND e2.team = os.team), FALSE) AS scored
            FROM o_starts os
            LEFT JOIN events e2 ON e2.game_id = os.game_id
                AND e2.event_number > os.event_number
                AND e2.event_number < COALESCE(
                    (SELECT MIN(e3.event_number) FROM events e3
                     WHERE e3.game_id = os.game_id AND e3.event_type = 1
                       AND e3.event_number > os.event_number), 9999999
                )
            GROUP BY os.event_id
        )
        SELECT
            lp.player_id,
            p.full_name,
            COUNT(*) AS possessions,
            ROUND(AVG(CASE WHEN oc.scored THEN 1.0 ELSE 0.0 END)::numeric, 4) AS hold_rate
        FROM line_players lp
        JOIN outcomes oc ON oc.event_id = lp.event_id
        JOIN players p ON p.player_id = lp.player_id
        WHERE lp.line_type = 'O'
        GROUP BY lp.player_id, p.full_name
        HAVING COUNT(*) >= GREATEST(5, (SELECT COUNT(*) * 0.20 FROM o_starts)::int)
        ORDER BY hold_rate DESC
    """, year_params_1)
    player_rows = cur.fetchall()
    top_players = [
        {"id": r['player_id'], "name": r['full_name'],
         "hold_rate": float(r['hold_rate']), "possessions": r['possessions']}
        for r in player_rows[:5]
    ]

    # Top 5 pair synergies for this team
    cur.execute(f"""
        WITH o_starts AS (
            SELECT e.event_id, e.game_id, e.event_number, e.team
            FROM events e WHERE e.event_type = 2 AND e.team = %s {year_filter}
        ),
        outcomes AS (
            SELECT os.event_id,
                COALESCE(BOOL_OR(e2.event_type = 19 AND e2.team = os.team), FALSE) AS scored
            FROM o_starts os
            LEFT JOIN events e2 ON e2.game_id = os.game_id
                AND e2.event_number > os.event_number
                AND e2.event_number < COALESCE(
                    (SELECT MIN(e3.event_number) FROM events e3
                     WHERE e3.game_id = os.game_id AND e3.event_type = 1
                       AND e3.event_number > os.event_number), 9999999
                )
            GROUP BY os.event_id
        ),
        player_rates AS (
            SELECT lp.player_id,
                AVG(CASE WHEN oc.scored THEN 1.0 ELSE 0.0 END) AS hold_rate
            FROM line_players lp
            JOIN outcomes oc ON oc.event_id = lp.event_id
            WHERE lp.line_type = 'O'
            GROUP BY lp.player_id
            HAVING COUNT(*) >= GREATEST(5, (SELECT COUNT(*) * 0.20 FROM o_starts)::int)
        ),
        pair_stats AS (
            SELECT lp1.player_id AS p1, lp2.player_id AS p2,
                COUNT(*) AS shared_poss,
                AVG(CASE WHEN oc.scored THEN 1.0 ELSE 0.0 END) AS combined_rate
            FROM line_players lp1
            JOIN line_players lp2 ON lp1.event_id = lp2.event_id
                AND lp1.line_type = 'O' AND lp2.line_type = 'O'
                AND lp1.player_id < lp2.player_id
            JOIN outcomes oc ON oc.event_id = lp1.event_id
            GROUP BY lp1.player_id, lp2.player_id
            HAVING COUNT(*) >= GREATEST(5, (SELECT COUNT(*) * 0.10 FROM o_starts)::int)
        )
        SELECT
            ps.p1, ps.p2,
            p1n.full_name AS p1_name, p2n.full_name AS p2_name,
            ps.shared_poss,
            ROUND(ps.combined_rate::numeric, 4) AS combined_rate,
            ROUND(pr1.hold_rate::numeric, 4) AS p1_rate,
            ROUND(pr2.hold_rate::numeric, 4) AS p2_rate,
            ROUND((ps.combined_rate - (pr1.hold_rate + pr2.hold_rate) / 2)::numeric, 4) AS synergy_delta
        FROM pair_stats ps
        JOIN player_rates pr1 ON pr1.player_id = ps.p1
        JOIN player_rates pr2 ON pr2.player_id = ps.p2
        JOIN players p1n ON p1n.player_id = ps.p1
        JOIN players p2n ON p2n.player_id = ps.p2
        ORDER BY synergy_delta DESC
        LIMIT 5
    """, year_params_1)
    synergy_rows = cur.fetchall()

    # Roster: all players who appeared on O or D line for this team (optionally in a given year)
    cur.execute(f"""
        SELECT
            lp.player_id,
            p.full_name,
            COUNT(*) FILTER (WHERE lp.line_type = 'O') AS o_appearances,
            COUNT(*) FILTER (WHERE lp.line_type = 'D') AS d_appearances
        FROM line_players lp
        JOIN events e ON e.event_id = lp.event_id
        JOIN players p ON p.player_id = lp.player_id
        WHERE e.team = %s {year_filter}
        GROUP BY lp.player_id, p.full_name
        ORDER BY (COUNT(*) FILTER (WHERE lp.line_type = 'O') + COUNT(*) FILTER (WHERE lp.line_type = 'D')) DESC
    """, year_params_1)
    roster_rows = cur.fetchall()
    roster = [
        {
            "id": r["player_id"],
            "name": r["full_name"],
            "o_appearances": int(r["o_appearances"] or 0),
            "d_appearances": int(r["d_appearances"] or 0),
        }
        for r in roster_rows
    ]

    # Past games for this team (filtered by year if provided)
    games_year_filter = "AND g.year = %s" if year else ""
    games_params = (team_id, team_id, year) if year else (team_id, team_id)
    cur.execute(f"""
        SELECT g.game_id, g.game_date, g.year, g.home_team_id, g.away_team_id,
               g.home_score, g.away_score
        FROM games g
        WHERE (g.home_team_id = %s OR g.away_team_id = %s)
          AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
          {games_year_filter}
        ORDER BY g.game_date DESC, g.game_id DESC
    """, games_params)
    games_rows = cur.fetchall()
    past_games = [
        {
            "game_id":      r["game_id"],
            "game_date":    str(r["game_date"]),
            "year":         r["year"],
            "home_team_id": r["home_team_id"],
            "away_team_id": r["away_team_id"],
            "home_score":   r["home_score"],
            "away_score":   r["away_score"],
            "won": (r["home_team_id"] == team_id and r["home_score"] > r["away_score"]) or
                   (r["away_team_id"] == team_id and r["away_score"] > r["home_score"]),
        }
        for r in games_rows
    ]

    cur.close()
    conn.close()

    return {
        "team_id":        team_row['team_id'],
        "team_name":      team_row['team_name'],
        "division":       team_row['division'],
        "available_years": available_years,
        "selected_year":  year,
        "record": {
            "wins":   record['wins'],
            "losses": record['losses'],
            "games":  record['total_games'],
        },
        "o_line_rate": round(o_line_rate, 4),
        "top_players": top_players,
        "roster": roster,
        "past_games": past_games,
        "top_synergies": [
            {
                "player1": {"id": r['p1'], "name": r['p1_name']},
                "player2": {"id": r['p2'], "name": r['p2_name']},
                "shared_possessions": r['shared_poss'],
                "combined_rate":      float(r['combined_rate']),
                "p1_rate":            float(r['p1_rate']),
                "p2_rate":            float(r['p2_rate']),
                "synergy_delta":      float(r['synergy_delta']),
            }
            for r in synergy_rows
        ],
    }


@app.get("/embeddings/players")
def get_player_embeddings() -> Dict[str, Any]:
    """Get UMAP 2D projection of player embeddings."""
    if _umap_coordinates is None or _player_encoder is None or _cluster_labels is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    players = _player_encoder.classes_.tolist()
    player_stats = {p: _player_stats[p] for p in players if p in _player_stats}
    cluster_info = {str(k): v for k, v in _cluster_summaries.items()}

    # Resolve player IDs to full names
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT player_id, full_name FROM players WHERE player_id = ANY(%s)", (players,))
    name_map = {r["player_id"]: r["full_name"] for r in cur.fetchall()}
    cur.close()
    conn.close()

    return {
        "players": players,
        "coordinates": _umap_coordinates.tolist(),
        "clusters": _cluster_labels.tolist(),
        "player_stats": player_stats,
        "cluster_summaries": cluster_info,
        "name_map": name_map,
    }


@app.post("/embeddings/recompute")
def recompute_embeddings():
    """Force recompute of UMAP + KMeans and update the DB cache."""
    global _umap_coordinates, _cluster_labels, _cluster_summaries
    if _player_encoder is None:
        raise HTTPException(status_code=503, detail="Player encoder not loaded")

    players_list = _player_encoder.classes_.tolist()
    stats_matrix = []
    for p in players_list:
        if p in _player_stats:
            s = _player_stats[p]
            stats_matrix.append([s[f] for f in STATS_FEATURES])
        else:
            stats_matrix.append([0.0] * len(STATS_FEATURES))

    stats_array = np.array(stats_matrix)
    scaler = StandardScaler()
    stats_normalized = scaler.fit_transform(stats_array)

    import umap as _umap
    reducer = _umap.UMAP(n_neighbors=15, min_dist=0.01, random_state=42)
    new_coords = reducer.fit_transform(stats_normalized)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    new_labels = kmeans.fit_predict(stats_normalized)

    _umap_coordinates = new_coords
    _cluster_labels = new_labels
    _cluster_summaries = {}
    for cluster_id in set(new_labels.tolist()):
        cluster_players = [players_list[i] for i, c in enumerate(new_labels) if c == cluster_id]
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

    _write_umap_cache(players_list, new_coords, new_labels)
    return {"status": "ok", "players": len(players_list), "clusters": len(_cluster_summaries)}


@app.get("/predict/throws")
def predict_throws(
    player: str = Query(..., description="Player ID"),
    x: float = Query(..., description="Thrower X position (-25 to 25)"),
    y: float = Query(..., description="Thrower Y position (0 to 120)"),
    grid_size: int = Query(30, ge=10, le=200, description="Grid resolution"),
) -> Dict[str, Any]:
    """Predict throw distribution for a player at a field position."""
    if not _ensure_flow_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    import torch

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


_THROW_CACHE_ORIGIN_X = np.linspace(-25, 25, 24)
_THROW_CACHE_ORIGIN_Y = np.linspace(0, 120, 20)

@app.get("/predict/throws/batch")
def predict_throws_batch(
    player: str = Query(..., description="Player ID"),
    grid_cells_x: int = Query(10, ge=3, le=20, description="Number of grid cells across field width"),
    grid_cells_y: int = Query(12, ge=3, le=24, description="Number of grid cells across field length"),
    heatmap_resolution: int = Query(30, ge=10, le=200, description="Resolution of each heatmap"),
) -> Dict[str, Any]:
    """Return throw density heatmaps from DB cache, falling back to live inference."""

    # Try DB cache first
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT origin_xi, origin_yi, origin_x, origin_y, grid FROM throw_heatmap_cache WHERE player_id=%s ORDER BY origin_xi, origin_yi",
            (player,)
        )
        rows = cur.fetchall()
        conn.close()
        if rows:
            print(f"[cache hit] {player}: {len(rows)} rows")
            results = {}
            x_positions_set = sorted(set(r["origin_x"] for r in rows))
            y_positions_set = sorted(set(r["origin_y"] for r in rows))
            for r in rows:
                results[f"{r['origin_xi']},{r['origin_yi']}"] = base64.b64encode(bytes(r["grid"])).decode()
            return {
                "grids": results,
                "x_positions": x_positions_set,
                "y_positions": y_positions_set,
                "extent": [-25, 25, 0, 120],
            }
        else:
            print(f"[cache miss] {player}: no rows found, falling back to inference")
    except Exception as e:
        print(f"[cache error] {player}: {e}, falling back to inference")

    # Fall back to live inference
    if not _ensure_flow_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")
    import torch

    if player not in _player_encoder.classes_:
        raise HTTPException(status_code=404, detail=f"Player '{player}' not found in model")

    player_encoded = int(_player_encoder.transform([player])[0])
    x_positions = np.linspace(-25, 25, grid_cells_x)
    y_positions = np.linspace(0, 120, grid_cells_y)
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
                ctx = torch.FloatTensor([[player_encoded, x_norm, y_norm]])
                ctx_feat = _context_net(ctx).expand(grid_points.shape[0], -1)
                log_probs = _flow.log_prob(grid_points, context=ctx_feat)
                probs = torch.exp(log_probs).numpy()
                results[f"{xi},{yi}"] = probs.reshape(len(y_bins), len(x_bins)).tolist()
    return {
        "grids": results,
        "x_positions": x_positions.tolist(),
        "y_positions": y_positions.tolist(),
        "extent": [-25, 25, 0, 120],
    }


_TURNOVER_CACHE_ORIGIN_X = np.linspace(-25, 25, 24)
_TURNOVER_CACHE_ORIGIN_Y = np.linspace(0, 120, 20)

def _nearest_cache_origin(x: float, y: float):
    """Return (xi, yi) of the nearest precomputed origin in the 24×20 grid."""
    xi = int(np.argmin(np.abs(_TURNOVER_CACHE_ORIGIN_X - x)))
    yi = int(np.argmin(np.abs(_TURNOVER_CACHE_ORIGIN_Y - y)))
    return xi, yi


@app.get("/predict/turnovers/batch")
def predict_turnovers_batch() -> Dict[str, Any]:
    """Return all precomputed turnover heatmaps from DB cache in one shot."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT origin_xi, origin_yi, origin_x, origin_y, grid FROM turnover_heatmap_cache ORDER BY origin_xi, origin_yi")
        rows = cur.fetchall()
        conn.close()
        if rows:
            x_positions = sorted(set(float(r["origin_x"]) for r in rows))
            y_positions = sorted(set(float(r["origin_y"]) for r in rows))
            grids = {}
            for r in rows:
                grids[f"{r['origin_xi']},{r['origin_yi']}"] = base64.b64encode(bytes(r["grid"])).decode()
            return {"grids": grids, "x_positions": x_positions, "y_positions": y_positions, "extent": [-25, 25, 0, 120]}
    except Exception as e:
        print(f"[turnovers batch cache error] {e}")
    raise HTTPException(status_code=503, detail="Turnover cache not available")


@app.get("/predict/blocks/batch")
def predict_blocks_batch() -> Dict[str, Any]:
    """Return all precomputed block heatmaps from DB cache in one shot."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT origin_xi, origin_yi, origin_x, origin_y, grid FROM block_heatmap_cache ORDER BY origin_xi, origin_yi")
        rows = cur.fetchall()
        conn.close()
        if rows:
            x_positions = sorted(set(float(r["origin_x"]) for r in rows))
            y_positions = sorted(set(float(r["origin_y"]) for r in rows))
            grids = {}
            for r in rows:
                grids[f"{r['origin_xi']},{r['origin_yi']}"] = base64.b64encode(bytes(r["grid"])).decode()
            return {"grids": grids, "x_positions": x_positions, "y_positions": y_positions, "extent": [-25, 25, 0, 120]}
    except Exception as e:
        print(f"[blocks batch cache error] {e}")
    raise HTTPException(status_code=503, detail="Block cache not available")


@app.get("/predict/turnovers")
def predict_turnovers(
    x: float = Query(..., description="Thrower X position (-25 to 25)"),
    y: float = Query(..., description="Thrower Y position (0 to 120)"),
    grid_size: int = Query(30, ge=10, le=200, description="Grid resolution"),
) -> Dict[str, Any]:
    """Predict turnover destination distribution from a field position."""
    xi, yi = _nearest_cache_origin(x, y)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT grid FROM turnover_heatmap_cache WHERE origin_xi=%s AND origin_yi=%s",
            (xi, yi)
        )
        row = cur.fetchone()
        conn.close()
        if row:
            grid = np.frombuffer(bytes(row["grid"]), dtype=np.float16).reshape(120, 100).astype(np.float32)
            return {"grid": grid.tolist(), "extent": [-25, 25, 0, 120]}
    except Exception as e:
        print(f"[turnover cache error] {e}, falling back to inference")

    if not _ensure_turnover_flow_loaded():
        raise HTTPException(status_code=503, detail="Turnover model not loaded")
    import torch

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
    return {"grid": grid.tolist(), "extent": [-25, 25, 0, 120]}


@app.get("/predict/blocks")
def predict_blocks(
    x: float = Query(..., description="Thrower X position (-25 to 25)"),
    y: float = Query(..., description="Thrower Y position (0 to 120)"),
    grid_size: int = Query(30, ge=10, le=200, description="Grid resolution"),
) -> Dict[str, Any]:
    """Predict block destination distribution from a field position."""
    xi, yi = _nearest_cache_origin(x, y)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT grid FROM block_heatmap_cache WHERE origin_xi=%s AND origin_yi=%s",
            (xi, yi)
        )
        row = cur.fetchone()
        conn.close()
        if row:
            grid = np.frombuffer(bytes(row["grid"]), dtype=np.float16).reshape(120, 100).astype(np.float32)
            return {"grid": grid.tolist(), "extent": [-25, 25, 0, 120]}
    except Exception as e:
        print(f"[block cache error] {e}, falling back to inference")

    if not _ensure_block_flow_loaded():
        raise HTTPException(status_code=503, detail="Block model not loaded")
    import torch

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
    return {"grid": grid.tolist(), "extent": [-25, 25, 0, 120]}


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
            return {
                "grid": np.zeros((grid_y, grid_x)).tolist(),
                "total_throws": 0,
                "total_turnovers": 0,
                "extent": [-25, 25, 0, 120],
            }

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


def _epv_predict_batch(grid_points: np.ndarray) -> np.ndarray:
    """Run EPV NN inference on a batch of feature rows."""
    import torch
    scaled = _epv_scaler.transform(grid_points)
    _epv_nn.eval()
    with torch.no_grad():
        return _epv_nn(torch.FloatTensor(scaled)).numpy()


def _epv_run_inference(throw_idx: int, team: Optional[str], quarter: Optional[int]) -> list:
    """Compute EPV grid via live inference. Returns 60×25 list of lists."""
    xs = np.linspace(-25, 25, 25)
    ys = np.linspace(0, 120, 60)
    xx, yy = np.meshgrid(xs, ys)
    n_points = xx.size

    quarters_to_avg = [quarter] if quarter is not None else [1, 2, 3, 4]
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
                np.zeros(n_points),
                np.zeros(n_points),
                np.full(n_points, tid),
                np.full(n_points, q),
            ]).astype(np.float32)
            all_probs[idx] = _epv_predict_batch(gp)
            idx += 1

    grid = all_probs.mean(axis=0).reshape(60, 25)
    return [[round(float(v), 4) for v in row] for row in grid]


@app.get("/epv/heatmap")
def get_epv_heatmap(
    throw_idx: int = Query(1, ge=1, le=10, description="Throw number within possession (1-10)"),
    team: Optional[str] = Query(None, description="Team name for team-specific EPV (omit for league average)"),
    quarter: Optional[int] = Query(None, ge=1, le=4, description="Game quarter 1-4 (omit for average over all quarters)"),
) -> Dict[str, Any]:
    """Return EPV probability grid over the field. Shape: 60×25, values in [0,1]."""
    team_key = team or ''
    quarter_key = quarter or 0
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT grid FROM epv_heatmap_cache WHERE throw_idx=%s AND team=%s AND quarter=%s",
            (throw_idx, team_key, quarter_key)
        )
        row = cur.fetchone()
        conn.close()
        if row:
            grid = np.frombuffer(bytes(row["grid"]), dtype=np.float16).reshape(60, 25).astype(np.float32)
            return {"grid": [[round(float(v), 4) for v in r] for r in grid], "extent": [-25.0, 25.0, 0.0, 120.0], "throw_idx": throw_idx}
    except Exception as e:
        print(f"[epv cache error] {e}, falling back to inference")

    if not _ensure_epv_loaded():
        raise HTTPException(status_code=503, detail="EPV model not loaded")
    try:
        return {"grid": _epv_run_inference(throw_idx, team, quarter), "extent": [-25.0, 25.0, 0.0, 120.0], "throw_idx": throw_idx}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Line Synergy Endpoints
# ---------------------------------------------------------------------------

@app.get("/synergy/players")
def get_synergy_players():
    """All players who appear on O-lines, with full names."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT lp.player_id, COALESCE(p.full_name, lp.player_id) as full_name
        FROM line_players lp
        LEFT JOIN players p ON p.player_id = lp.player_id
        WHERE lp.line_type = 'O'
        ORDER BY full_name
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"id": r["player_id"], "name": r["full_name"]} for r in rows]


def _scoring_rate_for_player(cur, player_id: str) -> float:
    """Compute O-line scoring rate for a single player."""
    cur.execute("""
        WITH p_o_starts AS (
            SELECT lp.event_id, e.game_id, e.event_number, e.team
            FROM line_players lp
            JOIN events e ON e.event_id = lp.event_id
            WHERE lp.player_id = %(pid)s AND lp.line_type = 'O'
        ),
        outcomes AS (
            SELECT os.event_id,
                COALESCE(BOOL_OR(e2.event_type = 19 AND e2.team = os.team), FALSE) AS scored
            FROM p_o_starts os
            LEFT JOIN events e2 ON e2.game_id = os.game_id
                AND e2.event_number > os.event_number
                AND e2.event_number < COALESCE(
                    (SELECT MIN(e3.event_number) FROM events e3
                     WHERE e3.game_id = os.game_id AND e3.event_type = 1
                       AND e3.event_number > os.event_number),
                    9999999
                )
            GROUP BY os.event_id
        )
        SELECT ROUND(AVG(CASE WHEN scored THEN 1.0 ELSE 0.0 END)::numeric, 4) as scoring_rate
        FROM outcomes
    """, {"pid": player_id})
    row = cur.fetchone()
    return float(row["scoring_rate"] or 0)


@app.get("/synergy/pair")
def get_synergy_pair(player1: str = Query(...), player2: str = Query(...)):
    """Compute synergy metrics between two players on O-lines."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Validate both players exist
    cur.execute("""
        SELECT COUNT(DISTINCT player_id) as cnt FROM line_players
        WHERE player_id IN (%(p1)s, %(p2)s) AND line_type = 'O'
    """, {"p1": player1, "p2": player2})
    if cur.fetchone()["cnt"] < 2:
        cur.close(); conn.close()
        raise HTTPException(status_code=404, detail="One or both players not found in O-line data")

    # Shared possessions + combined scoring rate
    cur.execute("""
        WITH shared_o_starts AS (
            SELECT lp1.event_id, e.game_id, e.event_number, e.team
            FROM line_players lp1
            JOIN line_players lp2 ON lp2.event_id = lp1.event_id
                AND lp2.player_id = %(p2)s AND lp2.line_type = 'O'
            JOIN events e ON e.event_id = lp1.event_id
            WHERE lp1.player_id = %(p1)s AND lp1.line_type = 'O'
        ),
        outcomes AS (
            SELECT os.event_id,
                COALESCE(BOOL_OR(e2.event_type = 19 AND e2.team = os.team), FALSE) AS scored
            FROM shared_o_starts os
            LEFT JOIN events e2 ON e2.game_id = os.game_id
                AND e2.event_number > os.event_number
                AND e2.event_number < COALESCE(
                    (SELECT MIN(e3.event_number) FROM events e3
                     WHERE e3.game_id = os.game_id AND e3.event_type = 1
                       AND e3.event_number > os.event_number),
                    9999999
                )
            GROUP BY os.event_id
        )
        SELECT COUNT(*) as shared_possessions,
               ROUND(AVG(CASE WHEN scored THEN 1.0 ELSE 0.0 END)::numeric, 4) as combined_scoring_rate
        FROM outcomes
    """, {"p1": player1, "p2": player2})
    shared_row = cur.fetchone()
    shared_possessions = int(shared_row["shared_possessions"])
    combined_scoring_rate = float(shared_row["combined_scoring_rate"] or 0)

    p1_rate = _scoring_rate_for_player(cur, player1)
    p2_rate = _scoring_rate_for_player(cur, player2)
    synergy_delta = round(combined_scoring_rate - (p1_rate + p2_rate) / 2, 4)

    # Throw exchange
    cur.execute("""
        SELECT
            SUM(CASE WHEN thrower = %(p1)s AND receiver = %(p2)s THEN 1 ELSE 0 END) as p1_to_p2_total,
            SUM(CASE WHEN thrower = %(p1)s AND receiver = %(p2)s AND event_type IN (18,19) THEN 1 ELSE 0 END) as p1_to_p2_comp,
            SUM(CASE WHEN thrower = %(p2)s AND receiver = %(p1)s THEN 1 ELSE 0 END) as p2_to_p1_total,
            SUM(CASE WHEN thrower = %(p2)s AND receiver = %(p1)s AND event_type IN (18,19) THEN 1 ELSE 0 END) as p2_to_p1_comp
        FROM events
        WHERE event_type IN (18, 19, 20, 22)
          AND ((thrower = %(p1)s AND receiver = %(p2)s) OR (thrower = %(p2)s AND receiver = %(p1)s))
    """, {"p1": player1, "p2": player2})
    t = cur.fetchone()

    def safe_pct(comp, total):
        return round(comp / total, 4) if total > 0 else 0.0

    p1_to_p2_total = int(t["p1_to_p2_total"] or 0)
    p2_to_p1_total = int(t["p2_to_p1_total"] or 0)

    # Player names
    cur.execute("SELECT player_id, full_name FROM players WHERE player_id IN (%(p1)s, %(p2)s)",
                {"p1": player1, "p2": player2})
    names = {r["player_id"]: r["full_name"] for r in cur.fetchall()}
    cur.close()
    conn.close()

    return {
        "player1": {"id": player1, "name": names.get(player1, player1)},
        "player2": {"id": player2, "name": names.get(player2, player2)},
        "shared_possessions": shared_possessions,
        "combined_scoring_rate": combined_scoring_rate,
        "p1_scoring_rate": p1_rate,
        "p2_scoring_rate": p2_rate,
        "synergy_delta": synergy_delta,
        "p1_to_p2": {"count": p1_to_p2_total, "completion_pct": safe_pct(int(t["p1_to_p2_comp"] or 0), p1_to_p2_total)},
        "p2_to_p1": {"count": p2_to_p1_total, "completion_pct": safe_pct(int(t["p2_to_p1_comp"] or 0), p2_to_p1_total)},
    }


@app.get("/lineup/predict")
def predict_lineup(
    p1: str = Query(...), p2: str = Query(...), p3: str = Query(...),
    p4: str = Query(...), p5: str = Query(...), p6: str = Query(...),
    p7: str = Query(...),
):
    """Predict P(score) for a 7-player O-lineup."""
    if _lineup_xgb is None:
        raise HTTPException(status_code=503, detail="Lineup model not loaded")

    lineup = [p1, p2, p3, p4, p5, p6, p7]
    o_stats = [
        [_lineup_player_stats[pid][col] for col in _lineup_stat_cols]
        for pid in lineup if pid in _lineup_player_stats
    ]

    if not o_stats:
        raise HTTPException(status_code=400, detail="None of the players have stats data")

    o_arr = np.array(o_stats)
    # D-line unknown at inference time — use league averages (same as training fallback)
    d_arr = np.array([[_lineup_league_avg[col] for col in _lineup_stat_cols]])

    feats: List[float] = []
    for arr in [o_arr, d_arr]:
        for col_vals in arr.T:
            feats += [float(col_vals.mean()), float(col_vals.max()), float(col_vals.min()), float(col_vals.std())]

    feats.append(float(np.sum(o_arr[:, _lineup_stat_cols.index('completion_pct')] > _lineup_league_avg['completion_pct'])))
    feats.append(float(np.sum(o_arr[:, _lineup_stat_cols.index('huck_rate')] > _lineup_league_avg['huck_rate'])))
    feats.append(float(len(o_stats)))
    feats.append(0.0)   # n_d_players_with_stats = 0 (league avg used)

    # Pairwise line familiarity (21 pairs in a 7-player lineup)
    from itertools import combinations
    pair_counts = []
    for pid1, pid2 in combinations(sorted(lineup), 2):
        key = (pid1, pid2) if pid1 < pid2 else (pid2, pid1)
        pair_counts.append(_lineup_pair_familiarity.get(key, 0))
    feats += [
        float(np.mean(pair_counts)) if pair_counts else 0.0,
        float(np.min(pair_counts)) if pair_counts else 0.0,
        float(np.max(pair_counts)) if pair_counts else 0.0,
        float(sum(1 for c in pair_counts if c > 0)),
    ]

    feats += [0.0, 0.0, 0.0]  # score_diff, total_score, quarter (unknown at inference)

    prob = float(_lineup_xgb.predict_proba(np.array([feats], dtype=np.float32))[0, 1])

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT player_id, full_name FROM players WHERE player_id = ANY(%s)", (lineup,))
    names = {r["player_id"]: r["full_name"] for r in cur.fetchall()}
    cur.close()
    conn.close()

    return {
        "probability": round(prob, 4),
        "players": [{"id": pid, "name": names.get(pid, pid)} for pid in lineup],
        "known_players": len(o_stats),
    }


def _combined_scoring_rate_for_pair(cur, p1: str, p2: str):
    """Returns (shared_possessions, combined_scoring_rate) for an O-line pair."""
    cur.execute("""
        WITH shared_o_starts AS (
            SELECT lp1.event_id, e.game_id, e.event_number, e.team
            FROM line_players lp1
            JOIN line_players lp2 ON lp2.event_id = lp1.event_id
                AND lp2.player_id = %(p2)s AND lp2.line_type = 'O'
            JOIN events e ON e.event_id = lp1.event_id
            WHERE lp1.player_id = %(p1)s AND lp1.line_type = 'O'
        ),
        outcomes AS (
            SELECT os.event_id,
                COALESCE(BOOL_OR(e2.event_type = 19 AND e2.team = os.team), FALSE) AS scored
            FROM shared_o_starts os
            LEFT JOIN events e2 ON e2.game_id = os.game_id
                AND e2.event_number > os.event_number
                AND e2.event_number < COALESCE(
                    (SELECT MIN(e3.event_number) FROM events e3
                     WHERE e3.game_id = os.game_id AND e3.event_type = 1
                       AND e3.event_number > os.event_number),
                    9999999
                )
            GROUP BY os.event_id
        )
        SELECT COUNT(*) as shared_possessions,
               ROUND(AVG(CASE WHEN scored THEN 1.0 ELSE 0.0 END)::numeric, 4) as combined_rate
        FROM outcomes
    """, {"p1": p1, "p2": p2})
    row = cur.fetchone()
    return int(row["shared_possessions"]), float(row["combined_rate"] or 0)


@app.get("/player/{player_id}")
def get_player(player_id: str, year: Optional[int] = None):
    """Per-player stats: season table, career totals, top connections, synergy partners."""
    conn = get_db_connection()
    cur = conn.cursor()

    # Player info
    cur.execute("SELECT player_id, full_name FROM players WHERE player_id = %s", (player_id,))
    player_row = cur.fetchone()
    if not player_row:
        cur.close(); conn.close()
        raise HTTPException(status_code=404, detail=f"Player '{player_id}' not found")
    player_name = player_row["full_name"]

    # Available years
    cur.execute("""
        SELECT DISTINCT year FROM events
        WHERE (thrower = %s OR receiver = %s) AND year IS NOT NULL
        ORDER BY year
    """, (player_id, player_id))
    available_years = [r["year"] for r in cur.fetchall()]

    yf_plain = "AND year = %s" if year else ""
    yf_e = "AND e.year = %s" if year else ""
    yp1 = (player_id, year) if year else (player_id,)

    def _fetch_seasons(yf_p, yf_ev, params):
        """Fetch and merge per-season stats rows. Used for both all-time and filtered."""
        cur.execute(f"""
            SELECT year, team,
                COUNT(*) FILTER (WHERE event_type IN (18,19,20,22)) AS throw_attempts,
                COUNT(*) FILTER (WHERE event_type IN (18,19)) AS completions,
                COUNT(*) FILTER (WHERE event_type = 19) AS assists,
                COUNT(*) FILTER (WHERE event_type = 22) AS turnovers,
                COUNT(*) FILTER (WHERE event_type IN (18,19,20) AND receiver_x IS NOT NULL AND receiver_y IS NOT NULL
                    AND SQRT(POWER(receiver_x - thrower_x, 2) + POWER(receiver_y - thrower_y, 2)) >= 40) AS huck_completions,
                COUNT(*) FILTER (WHERE event_type IN (18,19,20,22)
                    AND COALESCE(receiver_x, turnover_x) IS NOT NULL AND COALESCE(receiver_y, turnover_y) IS NOT NULL
                    AND SQRT(POWER(COALESCE(receiver_x, turnover_x) - thrower_x, 2) + POWER(COALESCE(receiver_y, turnover_y) - thrower_y, 2)) >= 40) AS huck_attempts,
                ROUND(AVG(CASE WHEN event_type IN (18,19) AND receiver_x IS NOT NULL
                    THEN SQRT(POWER(receiver_x-thrower_x,2)+POWER(receiver_y-thrower_y,2)) END)::numeric, 1) AS avg_throw_dist,
                ROUND(AVG(CASE WHEN event_type IN (18,19) AND receiver_y IS NOT NULL
                    THEN receiver_y - thrower_y END)::numeric, 1) AS avg_throw_depth
            FROM events
            WHERE thrower = %s AND event_type IN (18,19,20,22) {yf_p}
            GROUP BY year, team ORDER BY year
        """, params)
        t_rows: Dict[int, Any] = {}
        for r in cur.fetchall():
            yr = r["year"]
            if yr not in t_rows or (r.get("throw_attempts") or 0) > (t_rows[yr].get("throw_attempts") or 0):
                t_rows[yr] = dict(r)

        cur.execute(f"""
            SELECT year, COUNT(*) AS catches,
                COUNT(*) FILTER (WHERE event_type = 19) AS goals_caught
            FROM events
            WHERE receiver = %s AND event_type IN (18,19) {yf_p}
            GROUP BY year ORDER BY year
        """, params)
        c_rows = {r["year"]: dict(r) for r in cur.fetchall()}

        cur.execute(f"""
            SELECT e.year,
                COUNT(DISTINCT lp.event_id) FILTER (WHERE lp.line_type = 'O') AS o_possessions,
                COUNT(DISTINCT lp.event_id) FILTER (WHERE lp.line_type = 'D') AS d_possessions
            FROM line_players lp
            JOIN events e ON e.event_id = lp.event_id
            WHERE lp.player_id = %s {yf_ev}
            GROUP BY e.year ORDER BY e.year
        """, params)
        p_rows = {r["year"]: dict(r) for r in cur.fetchall()}

        cur.execute(f"""
            WITH o_starts AS (
                SELECT e.event_id, e.game_id, e.event_number, e.team, e.year
                FROM events e
                JOIN line_players lp ON lp.event_id = e.event_id
                    AND lp.line_type = 'O' AND lp.player_id = %s
                WHERE e.event_type = 2 {yf_ev}
            ),
            outcomes AS (
                SELECT os.event_id, os.year,
                    COALESCE(BOOL_OR(e2.event_type = 19 AND e2.team = os.team), FALSE) AS scored
                FROM o_starts os
                LEFT JOIN events e2 ON e2.game_id = os.game_id
                    AND e2.event_number > os.event_number
                    AND e2.event_number < COALESCE(
                        (SELECT MIN(e3.event_number) FROM events e3
                         WHERE e3.game_id = os.game_id AND e3.event_type = 1
                           AND e3.event_number > os.event_number),
                        9999999
                    )
                GROUP BY os.event_id, os.year
            )
            SELECT year,
                ROUND(AVG(CASE WHEN scored THEN 1.0 ELSE 0.0 END)::numeric, 4) AS o_hold_rate
            FROM outcomes
            GROUP BY year ORDER BY year
        """, params)
        h_rows = {r["year"]: float(r["o_hold_rate"] or 0) for r in cur.fetchall()}

        cur.execute(f"""
            SELECT year, COUNT(*) AS blocks
            FROM events
            WHERE defender = %s AND event_type = 11 {yf_p}
            GROUP BY year ORDER BY year
        """, params)
        b_rows = {r["year"]: int(r["blocks"]) for r in cur.fetchall()}

        cur.execute(f"""
            SELECT year, COUNT(*) AS drops
            FROM events
            WHERE receiver = %s AND event_type = 20 {yf_p}
            GROUP BY year ORDER BY year
        """, params)
        d_rows = {r["year"]: int(r["drops"]) for r in cur.fetchall()}

        cur.execute(f"""
            SELECT e.year, COUNT(DISTINCT e.game_id) AS games
            FROM line_players lp
            JOIN events e ON e.event_id = lp.event_id
            WHERE lp.player_id = %s {yf_ev}
            GROUP BY e.year ORDER BY e.year
        """, params)
        g_rows = {r["year"]: int(r["games"]) for r in cur.fetchall()}

        all_yrs = sorted(set(list(t_rows.keys()) + list(c_rows.keys()) + list(p_rows.keys())))
        result = []
        for yr in all_yrs:
            t = t_rows.get(yr, {}); c = c_rows.get(yr, {}); p = p_rows.get(yr, {})
            ta = int(t.get("throw_attempts") or 0); comp = int(t.get("completions") or 0)
            ha = int(t.get("huck_attempts") or 0); hc = int(t.get("huck_completions") or 0)
            goals = int(c.get("goals_caught") or 0)
            assists = int(t.get("assists") or 0)
            turnovers = int(t.get("turnovers") or 0)
            blocks = b_rows.get(yr, 0)
            drops = d_rows.get(yr, 0)
            result.append({
                "year": yr, "team": t.get("team") or "",
                "games": g_rows.get(yr, 0),
                "o_possessions": int(p.get("o_possessions") or 0),
                "d_possessions": int(p.get("d_possessions") or 0),
                "throw_attempts": ta, "completions": comp,
                "completion_pct": round(comp / ta, 4) if ta > 0 else 0.0,
                "assists": assists,
                "goals": goals,
                "turnovers": turnovers,
                "drops": drops,
                "plus_minus": goals + assists + blocks - drops - turnovers,
                "huck_attempts": ha, "huck_completions": hc,
                "huck_pct": round(hc / ha, 4) if ha > 0 else 0.0,
                "avg_throw_dist": float(t.get("avg_throw_dist") or 0),
                "avg_throw_depth": float(t.get("avg_throw_depth") or 0),
                "catches": int(c.get("catches") or 0),
                "blocks": blocks,
                "o_hold_rate": h_rows.get(yr, 0.0),
            })
        return result

    # Always fetch all seasons for career totals; filter separately for display
    all_seasons = _fetch_seasons("", "", (player_id,))
    seasons = [s for s in all_seasons if s["year"] == year] if year else all_seasons

    # Career totals always computed from all seasons
    def _career_from_seasons(ss):
        total_ta = sum(s["throw_attempts"] for s in ss)
        total_comp = sum(s["completions"] for s in ss)
        total_ha = sum(s["huck_attempts"] for s in ss)
        total_hc = sum(s["huck_completions"] for s in ss)
        total_opp = sum(s["o_possessions"] for s in ss)
        weighted_dist = sum(s["avg_throw_dist"] * s["completions"] for s in ss if s["completions"] > 0)
        weighted_depth = sum(s["avg_throw_depth"] * s["completions"] for s in ss if s["completions"] > 0)
        weighted_hold = sum(s["o_hold_rate"] * s["o_possessions"] for s in ss if s["o_possessions"] > 0)
        total_goals = sum(s["goals"] for s in ss)
        total_assists = sum(s["assists"] for s in ss)
        total_blocks = sum(s["blocks"] for s in ss)
        total_drops = sum(s["drops"] for s in ss)
        total_turnovers = sum(s["turnovers"] for s in ss)
        return {
            "year": 0, "team": "Career",
            "games": sum(s["games"] for s in ss),
            "o_possessions": total_opp,
            "d_possessions": sum(s["d_possessions"] for s in ss),
            "throw_attempts": total_ta, "completions": total_comp,
            "completion_pct": round(total_comp / total_ta, 4) if total_ta > 0 else 0.0,
            "assists": total_assists,
            "goals": total_goals,
            "turnovers": total_turnovers,
            "drops": total_drops,
            "plus_minus": total_goals + total_assists + total_blocks - total_drops - total_turnovers,
            "huck_attempts": total_ha, "huck_completions": total_hc,
            "huck_pct": round(total_hc / total_ha, 4) if total_ha > 0 else 0.0,
            "avg_throw_dist": round(weighted_dist / total_comp, 1) if total_comp > 0 else 0.0,
            "avg_throw_depth": round(weighted_depth / total_comp, 1) if total_comp > 0 else 0.0,
            "catches": sum(s["catches"] for s in ss),
            "blocks": total_blocks,
            "o_hold_rate": round(weighted_hold / total_opp, 4) if total_opp > 0 else 0.0,
        }

    career = _career_from_seasons(all_seasons)

    # Top 5 targets (who this player throws to most)
    cur.execute(f"""
        SELECT e.receiver AS player_id, COALESCE(p.full_name, e.receiver) AS name,
            COUNT(*) AS count,
            ROUND(SUM(CASE WHEN e.event_type IN (18,19) THEN 1.0 ELSE 0.0 END) / COUNT(*), 3) AS completion_pct
        FROM events e
        LEFT JOIN players p ON p.player_id = e.receiver
        WHERE e.thrower = %s AND e.event_type IN (18,19,20) AND e.receiver IS NOT NULL {yf_e}
        GROUP BY e.receiver, p.full_name
        ORDER BY count DESC LIMIT 5
    """, yp1)
    top_targets = [
        {"id": r["player_id"], "name": r["name"], "count": int(r["count"]), "completion_pct": float(r["completion_pct"])}
        for r in cur.fetchall()
    ]

    # Top 5 throwers (who throws to this player most)
    cur.execute(f"""
        SELECT e.thrower AS player_id, COALESCE(p.full_name, e.thrower) AS name,
            COUNT(*) AS count,
            ROUND(SUM(CASE WHEN e.event_type IN (18,19) THEN 1.0 ELSE 0.0 END) / COUNT(*), 3) AS completion_pct
        FROM events e
        LEFT JOIN players p ON p.player_id = e.thrower
        WHERE e.receiver = %s AND e.event_type IN (18,19,20) {yf_e}
        GROUP BY e.thrower, p.full_name
        ORDER BY count DESC LIMIT 5
    """, yp1)
    top_throwers = [
        {"id": r["player_id"], "name": r["name"], "count": int(r["count"]), "completion_pct": float(r["completion_pct"])}
        for r in cur.fetchall()
    ]

    # Top synergy partners (all-time; top 10 by shared possessions → top 5 by delta)
    cur.execute("""
        SELECT lp2.player_id AS teammate_id, COALESCE(p.full_name, lp2.player_id) AS name,
            COUNT(DISTINCT lp2.event_id) AS shared_possessions
        FROM line_players lp1
        JOIN line_players lp2 ON lp2.event_id = lp1.event_id
            AND lp2.line_type = 'O' AND lp2.player_id != %s
        LEFT JOIN players p ON p.player_id = lp2.player_id
        WHERE lp1.player_id = %s AND lp1.line_type = 'O'
        GROUP BY lp2.player_id, p.full_name
        HAVING COUNT(DISTINCT lp2.event_id) >= 10
        ORDER BY shared_possessions DESC
        LIMIT 10
    """, (player_id, player_id))
    top_teammates = cur.fetchall()

    player_rate = _scoring_rate_for_player(cur, player_id)
    synergy_partners = []
    for tm in top_teammates:
        teammate_id = tm["teammate_id"]
        shared, combined_rate = _combined_scoring_rate_for_pair(cur, player_id, teammate_id)
        teammate_rate = _scoring_rate_for_player(cur, teammate_id)
        delta = round(combined_rate - (player_rate + teammate_rate) / 2, 4)
        synergy_partners.append({
            "id": teammate_id,
            "name": tm["name"],
            "shared_possessions": shared,
            "combined_rate": combined_rate,
            "synergy_delta": delta,
        })
    synergy_partners.sort(key=lambda x: x["synergy_delta"], reverse=True)
    synergy_partners = synergy_partners[:5]

    cur.close()
    conn.close()

    return {
        "player": {"id": player_id, "name": player_name},
        "available_years": available_years,
        "seasons": seasons,
        "career": career,
        "top_targets": top_targets,
        "top_throwers": top_throwers,
        "synergy_partners": synergy_partners,
    }


@app.get("/player/{player_id}/game-log")
def get_player_game_log(player_id: str, year: Optional[int] = None):
    """Per-game stats for a player, optionally filtered by year."""
    conn = get_db_connection()
    cur = conn.cursor()
    yf = "AND e.year = %(yr)s" if year else ""
    params: Dict[str, Any] = {"pid": player_id}
    if year:
        params["yr"] = year
    cur.execute(f"""
        SELECT
            e.game_id,
            g.game_date::text AS game_date,
            g.home_team_id, g.away_team_id,
            g.home_score, g.away_score,
            MAX(CASE WHEN e.thrower = %(pid)s OR e.receiver = %(pid)s THEN e.team END) AS team,
            COUNT(*) FILTER (WHERE e.thrower = %(pid)s AND e.event_type IN (18,19,20,22)) AS throw_attempts,
            COUNT(*) FILTER (WHERE e.thrower = %(pid)s AND e.event_type IN (18,19)) AS completions,
            COUNT(*) FILTER (WHERE e.thrower = %(pid)s AND e.event_type = 19) AS assists,
            COUNT(*) FILTER (WHERE e.thrower = %(pid)s AND e.event_type IN (20,22)) AS turnovers,
            COUNT(*) FILTER (WHERE e.receiver = %(pid)s AND e.event_type IN (18,19)) AS catches,
            COUNT(*) FILTER (WHERE e.receiver = %(pid)s AND e.event_type = 19) AS goals,
            COUNT(*) FILTER (WHERE e.receiver = %(pid)s AND e.event_type = 20) AS drops,
            COUNT(*) FILTER (WHERE e.defender = %(pid)s AND e.event_type = 11) AS blocks
        FROM events e
        JOIN games g ON g.game_id = e.game_id
        WHERE (e.thrower = %(pid)s OR e.receiver = %(pid)s OR e.defender = %(pid)s) {yf}
        GROUP BY e.game_id, g.game_date, g.home_team_id, g.away_team_id, g.home_score, g.away_score
        ORDER BY g.game_date, e.game_id
    """, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    result = []
    for r in rows:
        goals = int(r["goals"] or 0)
        assists = int(r["assists"] or 0)
        blocks = int(r["blocks"] or 0)
        drops = int(r["drops"] or 0)
        turnovers = int(r["turnovers"] or 0)
        result.append({
            "game_id": r["game_id"],
            "game_date": r["game_date"],
            "home_team_id": r["home_team_id"],
            "away_team_id": r["away_team_id"],
            "home_score": r["home_score"],
            "away_score": r["away_score"],
            "team": r["team"] or "",
            "throw_attempts": int(r["throw_attempts"] or 0),
            "completions": int(r["completions"] or 0),
            "assists": assists,
            "turnovers": turnovers,
            "catches": int(r["catches"] or 0),
            "goals": goals,
            "drops": drops,
            "blocks": blocks,
            "plus_minus": goals + assists + blocks - drops - turnovers,
        })
    return result


@app.get("/player/{player_id}/throw-tendencies")
def get_player_throw_tendencies(player_id: str, year: Optional[int] = None):
    """Throw direction tendencies for a player, binned into 16 compass sectors."""
    conn = get_db_connection()
    cur = conn.cursor()

    yf = "AND e.year = %s" if year else ""
    params = (player_id, year) if year else (player_id,)

    cur.execute(f"""
        SELECT
            CASE WHEN e.team = g.home_team_id THEN receiver_x - thrower_x
                 ELSE thrower_x - receiver_x END AS dx,
            CASE WHEN e.team = g.home_team_id THEN receiver_y - thrower_y
                 ELSE thrower_y - receiver_y END AS dy,
            SQRT(POWER(receiver_x - thrower_x, 2) + POWER(receiver_y - thrower_y, 2)) AS dist
        FROM events e
        JOIN games g ON g.game_id = e.game_id
        WHERE e.thrower = %s
          AND e.event_type IN (18, 19)
          AND e.thrower_x IS NOT NULL AND e.thrower_y IS NOT NULL
          AND e.receiver_x IS NOT NULL AND e.receiver_y IS NOT NULL
          {yf}
    """, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return {"bins": [], "total_throws": 0, "max_avg_dist": 0.0}

    N_BINS = 16
    BIN_DEG = 360.0 / N_BINS

    bins_count = [0] * N_BINS
    bins_dist_sum = [0.0] * N_BINS

    for row in rows:
        dx, dy, dist = float(row["dx"]), float(row["dy"]), float(row["dist"])
        if dist < 1.0:
            continue
        # atan2(dx, dy): 0 = forward/upfield, clockwise positive; result in [-π, π]
        angle_deg = float(np.degrees(np.arctan2(dx, dy))) % 360.0
        bin_idx = int(angle_deg / BIN_DEG) % N_BINS
        bins_count[bin_idx] += 1
        bins_dist_sum[bin_idx] += dist

    total = sum(bins_count)
    max_avg_dist = 0.0
    result_bins = []
    for i in range(N_BINS):
        cnt = bins_count[i]
        avg_dist = bins_dist_sum[i] / cnt if cnt > 0 else 0.0
        max_avg_dist = max(max_avg_dist, avg_dist)
        result_bins.append({
            "angle_deg": round(i * BIN_DEG, 2),
            "count": cnt,
            "pct": round(cnt / total, 4) if total > 0 else 0.0,
            "avg_dist": round(avg_dist, 1),
        })

    return {
        "bins": result_bins,
        "total_throws": total,
        "max_avg_dist": round(max_avg_dist, 1),
    }


@app.get("/player/{player_id}/block-types")
def get_player_block_types(player_id: str, year: Optional[int] = None):
    """Classify each block by the throw type it came from: huck / short / reset."""
    yf = "AND b.year = %s" if year else ""
    params = (player_id, year) if year else (player_id,)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            CASE
                WHEN SQRT(
                    POWER(COALESCE(prev.receiver_x, prev.turnover_x, prev.thrower_x) - prev.thrower_x, 2) +
                    POWER(COALESCE(prev.receiver_y, prev.turnover_y, prev.thrower_y) - prev.thrower_y, 2)
                ) >= 40 THEN 'huck'
                WHEN (COALESCE(prev.receiver_y, prev.turnover_y, prev.thrower_y) - prev.thrower_y) < 2 THEN 'reset'
                ELSE 'short'
            END AS block_type,
            COUNT(*) AS cnt
        FROM events b
        JOIN events prev
          ON prev.game_id = b.game_id
         AND prev.event_number = b.event_number - 1
        WHERE b.event_type = 11
          AND b.defender = %s
          AND prev.thrower_x IS NOT NULL {yf}
        GROUP BY block_type
    """, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    result = {"huck": 0, "short": 0, "reset": 0}
    for r in rows:
        if r["block_type"] in result:
            result[r["block_type"]] = int(r["cnt"])
    result["total"] = result["huck"] + result["short"] + result["reset"]
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
