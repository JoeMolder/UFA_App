from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

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


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "message": "UFA Analytics API",
        "status": "running",
        "version": "1.0.0"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
