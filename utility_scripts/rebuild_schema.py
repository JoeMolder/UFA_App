#!/usr/bin/env python3
"""
Rebuild database schema - drops all tables and recreates them.
Run this before re-ingesting data with the new coordinate averaging.
"""

import psycopg2

# Database connection configuration
DB_CONFIG = {
    'dbname': 'ufa_analytics',
    'user': 'joemolder',
    'password': '',
    'host': 'localhost',
    'port': 5432
}

def get_connection():
    """Create and return a database connection."""
    return psycopg2.connect(**DB_CONFIG)

def create_schema():
    """Create all database tables."""
    conn = get_connection()
    cur = conn.cursor()

    print("Dropping existing tables...")
    # Drop existing tables (careful!)
    cur.execute("""
        DROP TABLE IF EXISTS line_players CASCADE;
        DROP TABLE IF EXISTS events CASCADE;
        DROP TABLE IF EXISTS games CASCADE;
        DROP TABLE IF EXISTS players CASCADE;
        DROP TABLE IF EXISTS teams CASCADE;
    """)
    print("✅ Tables dropped")

    print("\nCreating games table...")
    # Create games table
    cur.execute("""
        CREATE TABLE games (
            game_id VARCHAR(50) PRIMARY KEY,
            home_team_id VARCHAR(50) NOT NULL,
            away_team_id VARCHAR(50) NOT NULL,
            home_score INT,
            away_score INT,
            game_date DATE,
            year INT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    print("Creating teams table...")
    # Create teams table
    cur.execute("""
        CREATE TABLE teams (
            team_id VARCHAR(50) PRIMARY KEY,
            team_name VARCHAR(100),
            division VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    print("Creating players table...")
    # Create players table
    cur.execute("""
        CREATE TABLE players (
            player_id VARCHAR(50) PRIMARY KEY,
            full_name VARCHAR(100),
            team_id VARCHAR(50),
            year INT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    print("Creating events table...")
    # Create events table
    cur.execute("""
        CREATE TABLE events (
            event_id SERIAL PRIMARY KEY,
            game_id VARCHAR(50) NOT NULL REFERENCES games(game_id),
            event_number INT NOT NULL,
            event_type INT NOT NULL,
            team VARCHAR(50),
            year INT NOT NULL,

            -- Time field (for types 1, 2)
            time INT,

            -- Pull fields (for types 7, 8, 9, 10)
            puller VARCHAR(50),
            pull_x DOUBLE PRECISION,
            pull_y DOUBLE PRECISION,
            pull_ms INT,

            -- Throw/Pass fields (for types 18, 19, 20, 21)
            thrower VARCHAR(50),
            thrower_x DOUBLE PRECISION,
            thrower_y DOUBLE PRECISION,
            receiver VARCHAR(50),
            receiver_x DOUBLE PRECISION,
            receiver_y DOUBLE PRECISION,

            -- Turnover location fields (for types 22, 23)
            turnover_x DOUBLE PRECISION,
            turnover_y DOUBLE PRECISION,

            -- Defender field (for types 11, 12)
            defender VARCHAR(50),

            -- Player field (for types 26, 27)
            player VARCHAR(50),

            -- Custom fields
            synthetic BOOLEAN DEFAULT FALSE,

            created_at TIMESTAMP DEFAULT NOW(),

            CONSTRAINT unique_event_position UNIQUE(game_id, event_number)
        );
    """)

    print("Creating indexes on events...")
    # Create indexes on events
    cur.execute("""
        CREATE INDEX idx_events_game_id ON events(game_id);
        CREATE INDEX idx_events_type ON events(event_type);
        CREATE INDEX idx_events_team ON events(team);
        CREATE INDEX idx_events_synthetic ON events(synthetic);
        CREATE INDEX idx_events_thrower ON events(thrower);
        CREATE INDEX idx_events_receiver ON events(receiver);
        CREATE INDEX idx_events_defender ON events(defender);
    """)

    print("Creating line_players table...")
    # Create line_players table
    cur.execute("""
        CREATE TABLE line_players (
            line_player_id SERIAL PRIMARY KEY,
            event_id INT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
            game_id VARCHAR(50) NOT NULL,
            player_id VARCHAR(50) NOT NULL,
            line_type CHAR(1),
            position_order INT,
            team VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW(),

            CONSTRAINT unique_player_per_event UNIQUE(event_id, player_id)
        );
    """)

    print("Creating indexes on line_players...")
    # Create indexes on line_players
    cur.execute("""
        CREATE INDEX idx_line_players_event ON line_players(event_id);
        CREATE INDEX idx_line_players_player ON line_players(player_id);
        CREATE INDEX idx_line_players_game ON line_players(game_id);
        CREATE INDEX idx_line_players_team ON line_players(team);
        CREATE INDEX idx_line_players_type ON line_players(line_type);
    """)

    conn.commit()
    cur.close()
    conn.close()

    print("\n" + "="*60)
    print("✅ Database schema created successfully!")
    print("="*60)
    print("\nNext step: Run data ingestion to populate with cleaned data")

if __name__ == "__main__":
    create_schema()
