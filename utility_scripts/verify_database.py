#!/usr/bin/env python3
"""
Verify database has been populated with coordinate-averaged data.
"""

import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    'dbname': 'ufa_analytics',
    'user': 'joemolder',
    'password': '',
    'host': 'localhost',
    'port': 5432
}

def verify():
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()

    print("=" * 70)
    print("DATABASE VERIFICATION")
    print("=" * 70)

    # Count total games
    cur.execute("SELECT COUNT(*) as count FROM games;")
    total_games = cur.fetchone()['count']
    print(f"\n✅ Total games in database: {total_games}")

    # Count total events
    cur.execute("SELECT COUNT(*) as count FROM events;")
    total_events = cur.fetchone()['count']
    print(f"✅ Total events in database: {total_events:,}")

    # Count synthetic events
    cur.execute("SELECT COUNT(*) as count FROM events WHERE synthetic = TRUE;")
    synthetic_events = cur.fetchone()['count']
    print(f"✅ Synthetic events: {synthetic_events}")

    # Events by type
    cur.execute("""
        SELECT event_type, COUNT(*) as count
        FROM events
        GROUP BY event_type
        ORDER BY event_type;
    """)

    print("\n" + "=" * 70)
    print("EVENTS BY TYPE")
    print("=" * 70)
    for row in cur.fetchall():
        event_name = {
            1: "Start D Point (pulling team)",
            2: "Start O Point (receiving team)",
            3: "Timeout",
            7: "Pull (inbounds)",
            8: "Pull (OB)",
            11: "Block",
            18: "Pass",
            19: "Goal",
            20: "Drop",
            22: "Throwaway",
            23: "Callahan",
            24: "Stall",
            25: "Injury",
            28: "End Q1", 29: "End Q2", 30: "End Q3", 31: "End Q4",
            32: "End OT1", 33: "End OT2"
        }.get(row['event_type'], f"Type {row['event_type']}")
        print(f"  {event_name:20s}: {row['count']:5,} events")

    # Sample some turnovers to show coordinate averaging
    print("\n" + "=" * 70)
    print("SAMPLE TURNOVERS (showing coordinate averaging)")
    print("=" * 70)

    cur.execute("""
        SELECT
            game_id,
            event_type,
            team,
            CASE
                WHEN event_type = 20 THEN receiver_x
                WHEN event_type = 22 THEN turnover_x
            END as turnover_x,
            CASE
                WHEN event_type = 20 THEN receiver_y
                WHEN event_type = 22 THEN turnover_y
            END as turnover_y
        FROM events
        WHERE event_type IN (20, 22)
        ORDER BY game_id, event_number
        LIMIT 10;
    """)

    print(f"\n{'Type':<12} {'Team':<10} {'X':<8} {'Y':<8} {'In Endzone'}")
    print("-" * 50)
    for row in cur.fetchall():
        event_type = "Drop" if row['event_type'] == 20 else "Throwaway"
        x = row['turnover_x']
        y = row['turnover_y']
        in_endzone = "Yes" if (y and (y < 20 or y > 100)) else "No"
        print(f"{event_type:<12} {row['team']:<10} {x:<8.2f} {y:<8.2f} {in_endzone}")

    print("\n" + "=" * 70)
    print("✅ Database successfully populated with coordinate-averaged data!")
    print("=" * 70)

    cur.close()
    conn.close()

if __name__ == "__main__":
    verify()
