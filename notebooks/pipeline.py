"""
UFA Analytics — Full database setup and ingest pipeline.
Creates the ufa_analytics database, builds the schema, and populates
it with all game data for 2021-2025.
"""

import getpass
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import psycopg2
from psycopg2.extras import execute_batch
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import requests
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

DB_CONFIG = {
    "dbname":   "ufa_analytics",
    "user":     getpass.getuser(),  # matches default Postgres role (your OS username)
    "password": "",
    "host":     "localhost",
    "port":     5432,
}

YEARS         = [2021, 2022, 2023, 2024, 2025, 2026]
SKIP_EXISTING = True   # set False to reprocess games already in the DB
API_SLEEP_S   = 0.1

KEEP_EVENT_TYPES = {
    1, 2, 3, 7, 8, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33,
}
LINE_EVENT_TYPES = {1, 2, 3, 25}

FIELD_X_MIN, FIELD_X_MAX = -25.0, 25.0
FIELD_Y_MIN, FIELD_Y_MAX =   0.0, 110.0

# ── DB helpers ────────────────────────────────────────────────────────────────

def get_connection():
    url = os.environ.get('DATABASE_URL')
    if url:
        return psycopg2.connect(url, sslmode='require')
    return psycopg2.connect(**DB_CONFIG)


def create_database():
    conn = psycopg2.connect(
        dbname="postgres",
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG["dbname"],))
    if cur.fetchone():
        print(f"  Database '{DB_CONFIG['dbname']}' already exists — skipping creation.")
    else:
        cur.execute(f"CREATE DATABASE {DB_CONFIG['dbname']}")
        print(f"  Created database '{DB_CONFIG['dbname']}'")
    cur.close()
    conn.close()


def create_schema():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        DROP TABLE IF EXISTS line_players CASCADE;
        DROP TABLE IF EXISTS events CASCADE;
        DROP TABLE IF EXISTS games CASCADE;
        DROP TABLE IF EXISTS players CASCADE;
        DROP TABLE IF EXISTS teams CASCADE;
    """)
    cur.execute("""
        CREATE TABLE games (
            game_id        VARCHAR(50) PRIMARY KEY,
            home_team_id   VARCHAR(50) NOT NULL,
            away_team_id   VARCHAR(50) NOT NULL,
            home_score     INT,
            away_score     INT,
            game_date      DATE,
            year           INT NOT NULL,
            created_at     TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE teams (
            team_id    VARCHAR(50) PRIMARY KEY,
            team_name  VARCHAR(100),
            division   VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE players (
            player_id  VARCHAR(50) PRIMARY KEY,
            full_name  VARCHAR(100),
            team_id    VARCHAR(50),
            year       INT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        CREATE TABLE events (
            event_id     SERIAL PRIMARY KEY,
            game_id      VARCHAR(50) NOT NULL REFERENCES games(game_id),
            event_number INT NOT NULL,
            event_type   INT NOT NULL,
            team         VARCHAR(50),
            year         INT NOT NULL,
            time         INT,
            puller       VARCHAR(50),
            pull_x       DOUBLE PRECISION,
            pull_y       DOUBLE PRECISION,
            pull_ms      INT,
            thrower      VARCHAR(50),
            thrower_x    DOUBLE PRECISION,
            thrower_y    DOUBLE PRECISION,
            receiver     VARCHAR(50),
            receiver_x   DOUBLE PRECISION,
            receiver_y   DOUBLE PRECISION,
            turnover_x   DOUBLE PRECISION,
            turnover_y   DOUBLE PRECISION,
            defender     VARCHAR(50),
            player       VARCHAR(50),
            synthetic    BOOLEAN DEFAULT FALSE,
            created_at   TIMESTAMP DEFAULT NOW(),
            CONSTRAINT unique_event_position UNIQUE(game_id, event_number)
        );
        CREATE INDEX idx_events_game_id   ON events(game_id);
        CREATE INDEX idx_events_type      ON events(event_type);
        CREATE INDEX idx_events_team      ON events(team);
        CREATE INDEX idx_events_thrower   ON events(thrower);
        CREATE INDEX idx_events_receiver  ON events(receiver);
        CREATE INDEX idx_events_defender  ON events(defender);
        CREATE INDEX idx_events_synthetic ON events(synthetic);
        CREATE TABLE line_players (
            line_player_id SERIAL PRIMARY KEY,
            event_id       INT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
            game_id        VARCHAR(50) NOT NULL,
            player_id      VARCHAR(50) NOT NULL,
            line_type      CHAR(1),
            position_order INT,
            team           VARCHAR(50) NOT NULL,
            created_at     TIMESTAMP DEFAULT NOW(),
            CONSTRAINT unique_player_per_event UNIQUE(event_id, player_id)
        );
        CREATE INDEX idx_line_players_event  ON line_players(event_id);
        CREATE INDEX idx_line_players_player ON line_players(player_id);
        CREATE INDEX idx_line_players_game   ON line_players(game_id);
        CREATE INDEX idx_line_players_team   ON line_players(team);
        CREATE INDEX idx_line_players_type   ON line_players(line_type);
    """)
    conn.commit()
    cur.close()
    conn.close()

# ── API helpers ───────────────────────────────────────────────────────────────

def get_teams_data(years="all"):
    try:
        r = requests.get("https://www.backend.ufastats.com/api/v1/teams", params={"years": years})
        r.raise_for_status()
        return [
            {"team_id": t.get("teamID"), "team_name": t.get("fullName"),
             "division": t.get("division", {}).get("name"), "year": t.get("year")}
            for t in r.json().get("data", [])
        ]
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching teams: {e}")
        return []


def get_players_data(years="all"):
    try:
        r = requests.get("https://www.backend.ufastats.com/api/v1/players", params={"years": years})
        r.raise_for_status()
        players = []
        for p in r.json().get("data", []):
            pid  = p.get("playerID")
            full = f"{p.get('firstName', '')} {p.get('lastName', '')}".strip()
            for team in p.get("teams", []):
                players.append({"player_id": pid, "full_name": full,
                                 "team_id": team.get("teamID"), "year": team.get("year")})
        return players
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching players: {e}")
        return []


def get_games(date, statuses="Final"):
    try:
        r = requests.get("https://www.backend.ufastats.com/api/v1/games",
                         params={"date": date, "statuses": statuses})
        r.raise_for_status()
        games = []
        for g in r.json().get("data", []):
            gid = g.get("gameID")
            games.append({
                "game_id": gid, "home_team_id": g.get("homeTeamID"),
                "away_team_id": g.get("awayTeamID"), "home_score": g.get("homeScore"),
                "away_score": g.get("awayScore"),
                "year": int(gid[:4]) if gid else None,
                "game_date": gid[:10] if gid else None,
            })
        return games
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching games: {e}")
        return []

# ── Data cleaning ─────────────────────────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _jitter_endzone_xy(rng, base_x=0.0, base_y=110.0, max_dx=5.0, max_dy=15.0):
    x = base_x + rng.uniform(-max_dx, max_dx)
    y = base_y + rng.uniform(-max_dy, max_dy)
    return float(_clamp(x, FIELD_X_MIN, FIELD_X_MAX)), float(_clamp(y, FIELD_Y_MIN, FIELD_Y_MAX))


def clean_data(game_ids):
    events_url = "https://www.backend.ufastats.com/api/v1/gameEvents"
    games_url  = "https://www.backend.ufastats.com/api/v1/games"
    try:
        r = requests.get(games_url, params={"gameIDs": ",".join(game_ids)})
        r.raise_for_status()
        team_id_map = {g["gameID"]: {"home": g["homeTeamID"], "away": g["awayTeamID"],
                                     "year": g["gameID"][:4]}
                       for g in r.json().get("data", [])}
    except requests.exceptions.RequestException as e:
        print(f"  Error fetching game meta: {e}")
        return None

    results = {}
    for gameID in game_ids:
        try:
            r = requests.get(events_url, params={"gameID": gameID})
            if r.status_code != 200 or not r.headers.get("Content-Type", "").startswith("application/json"):
                continue
            data = r.json()
            home_team_id = team_id_map.get(gameID, {}).get("home", "Unknown")
            away_team_id = team_id_map.get(gameID, {}).get("away", "Unknown")
            game_year    = int(team_id_map.get(gameID, {}).get("year", 0))

            home_events = data.get("data", {}).get("homeEvents", [])
            away_events = data.get("data", {}).get("awayEvents", [])
            for ev in home_events:
                ev["team"] = home_team_id; ev["year"] = game_year; ev["gameID"] = gameID
            for ev in away_events:
                ev["team"] = away_team_id; ev["year"] = game_year; ev["gameID"] = gameID

            TERM        = {15, 19, 28, 29, 30, 31, 32, 33}
            SWITCH_TYPES = {3, 5, 15, 19, 20, 22, 23, 24, 28, 29, 30, 31, 32, 33}
            homeIndex = awayIndex = switch = 0
            combinedStack = []

            while homeIndex < len(home_events) or awayIndex < len(away_events):
                homeStack, awayStack = [], []
                while homeIndex < len(home_events):
                    homeStack.append(home_events[homeIndex])
                    t = home_events[homeIndex]["type"]; homeIndex += 1
                    if t in TERM: break
                while awayIndex < len(away_events):
                    awayStack.append(away_events[awayIndex])
                    t = away_events[awayIndex]["type"]; awayIndex += 1
                    if t in TERM: break

                if homeStack and homeStack[0]["type"] == 1:
                    combinedStack.append(homeStack[0])
                    if awayStack: combinedStack.append(awayStack[0])
                    if len(homeStack) > 1: combinedStack.append(homeStack[1])
                    homeStackIndex = 2; awayStackIndex = 1; switch = 1
                else:
                    if awayStack: combinedStack.append(awayStack[0])
                    if homeStack: combinedStack.append(homeStack[0])
                    if len(awayStack) > 1: combinedStack.append(awayStack[1])
                    homeStackIndex = 1; awayStackIndex = 2; switch = 0

                while homeStackIndex < len(homeStack) or awayStackIndex < len(awayStack):
                    if switch == 0 and homeStackIndex < len(homeStack):
                        while homeStackIndex < len(homeStack):
                            ev = homeStack[homeStackIndex]; combinedStack.append(ev); homeStackIndex += 1
                            if ev["type"] in SWITCH_TYPES: switch = 1; break
                            elif ev["type"] == 17:
                                if awayStackIndex < len(awayStack) and awayStack[awayStackIndex]["type"] == 16:
                                    combinedStack.append(awayStack[awayStackIndex]); awayStackIndex += 1
                                else:
                                    combinedStack.append({"type": 16, "team": home_team_id, "year": game_year, "gameID": gameID})
                            elif ev["type"] == 16:
                                if awayStackIndex < len(awayStack) and awayStack[awayStackIndex]["type"] == 17:
                                    combinedStack.insert(-1, awayStack[awayStackIndex]); awayStackIndex += 1
                                else:
                                    combinedStack.insert(-1, {"type": 17, "team": away_team_id, "year": game_year, "gameID": gameID})
                                break
                    elif switch == 1 and awayStackIndex < len(awayStack):
                        while awayStackIndex < len(awayStack):
                            ev = awayStack[awayStackIndex]; combinedStack.append(ev); awayStackIndex += 1
                            if ev["type"] in SWITCH_TYPES: switch = 0; break
                            elif ev["type"] == 17:
                                if homeStackIndex < len(homeStack) and homeStack[homeStackIndex]["type"] == 16:
                                    combinedStack.append(homeStack[homeStackIndex]); homeStackIndex += 1
                                else:
                                    combinedStack.append({"type": 16, "team": home_team_id, "year": game_year, "gameID": gameID})
                            elif ev["type"] == 16:
                                if homeStackIndex < len(homeStack) and homeStack[homeStackIndex]["type"] == 17:
                                    combinedStack.insert(-1, homeStack[homeStackIndex]); homeStackIndex += 1
                                else:
                                    combinedStack.insert(-1, {"type": 17, "team": home_team_id, "year": game_year, "gameID": gameID})
                                break
                    else:
                        break

            results[gameID] = combinedStack
        except requests.exceptions.RequestException as e:
            print(f"  Error fetching events for {gameID}: {e}")

    return results


def insert_missing_turnovers(combined_events, *, seed=None):
    rng = random.Random(seed)
    out, i, n = [], 0, len(combined_events)
    while i < n:
        e = combined_events[i]; out.append(e)
        k = len(out) - 1
        while k > 0 and out[k].get("type") == 11 and out[k-1].get("type") == 28:
            out[k-1], out[k] = out[k], out[k-1]; k -= 1
        if e.get("type") == 18:
            j, saw_22, insert_pos = i + 1, False, None
            while j < n:
                t = combined_events[j].get("type")
                if t == 22: saw_22 = True; break
                if t == 28: insert_pos = len(out); break
                if t not in (11,): break
                j += 1
            if not saw_22 and insert_pos is not None and e.get("receiver"):
                tX, tY = _jitter_endzone_xy(rng)
                out.insert(insert_pos, {
                    "type": 22, "thrower": e.get("receiver"),
                    "throwerX": e.get("receiverX"), "throwerY": e.get("receiverY"),
                    "turnoverX": round(tX, 2), "turnoverY": round(tY, 2),
                    "team": e.get("team"), "year": e.get("year"),
                    "gameID": e.get("gameID"), "synthetic": True,
                })
        i += 1
    return out


def normalize_coordinates(combined_events, away_team_id):
    COORD_FIELDS = [("throwerX","throwerY"),("receiverX","receiverY"),
                    ("turnoverX","turnoverY"),("pullX","pullY")]
    events = [ev.copy() for ev in combined_events]
    for ev in events:
        if ev.get("team") != away_team_id: continue
        for xk, yk in COORD_FIELDS:
            if ev.get(xk) is not None: ev[xk] = round(-ev[xk], 2)
            if ev.get(yk) is not None: ev[yk] = round(120 - ev[yk], 2)
    return events


def average_turnover_coordinates(combined_events):
    events = [ev.copy() for ev in combined_events]
    i = 0
    while i < len(events):
        ev = events[i]; etype = ev.get("type")
        if etype in (20, 22):
            team = ev.get("team")
            if etype == 20: tx, ty, txk, tyk = ev.get("receiverX"), ev.get("receiverY"), "receiverX", "receiverY"
            else:           tx, ty, txk, tyk = ev.get("turnoverX"), ev.get("turnoverY"), "turnoverX", "turnoverY"
            if tx is not None and ty is not None and not (ty < 20 or ty > 100):
                j = i + 1
                while j < len(events):
                    nev = events[j]
                    if nev.get("team") != team and nev.get("throwerX") is not None and nev.get("throwerY") is not None:
                        ax = round((tx + nev["throwerX"]) / 2, 2); ay = round((ty + nev["throwerY"]) / 2, 2)
                        events[i][txk] = ax; events[i][tyk] = ay
                        events[j]["throwerX"] = ax; events[j]["throwerY"] = ay
                        break
                    j += 1
        i += 1
    return events

# ── DB insertion ──────────────────────────────────────────────────────────────

def insert_teams(teams):
    conn = get_connection(); cur = conn.cursor()
    execute_batch(cur, """
        INSERT INTO teams (team_id, team_name, division) VALUES (%s, %s, %s)
        ON CONFLICT (team_id) DO UPDATE SET team_name=EXCLUDED.team_name, division=EXCLUDED.division;
    """, [(t["team_id"], t["team_name"], t["division"]) for t in teams])
    conn.commit(); cur.close(); conn.close()


def insert_players(players):
    conn = get_connection(); cur = conn.cursor()
    execute_batch(cur, """
        INSERT INTO players (player_id, full_name, team_id, year) VALUES (%s, %s, %s, %s)
        ON CONFLICT (player_id) DO UPDATE SET full_name=EXCLUDED.full_name, team_id=EXCLUDED.team_id, year=EXCLUDED.year;
    """, [(p["player_id"], p["full_name"], p["team_id"], p["year"]) for p in players])
    conn.commit(); cur.close(); conn.close()


def insert_game_data(game_id, events, home_team_id, away_team_id, home_score, away_score, year):
    conn = get_connection(); cur = conn.cursor()

    cur.execute("""
        INSERT INTO games (game_id, home_team_id, away_team_id, home_score, away_score, game_date, year)
        VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (game_id) DO NOTHING;
    """, (game_id, home_team_id, away_team_id, home_score, away_score, game_id[:10], year))

    rows = []
    for idx, ev in enumerate(events, 1):
        if ev.get("type") not in KEEP_EVENT_TYPES: continue
        rows.append((game_id, idx, ev.get("type"), ev.get("team"), ev.get("year"),
                     ev.get("time"), ev.get("puller"), ev.get("pullX"), ev.get("pullY"), ev.get("pullMs"),
                     ev.get("thrower"), ev.get("throwerX"), ev.get("throwerY"),
                     ev.get("receiver"), ev.get("receiverX"), ev.get("receiverY"),
                     ev.get("turnoverX"), ev.get("turnoverY"),
                     ev.get("defender"), ev.get("player"), ev.get("synthetic", False)))
    execute_batch(cur, """
        INSERT INTO events (
            game_id, event_number, event_type, team, year, time,
            puller, pull_x, pull_y, pull_ms,
            thrower, thrower_x, thrower_y, receiver, receiver_x, receiver_y,
            turnover_x, turnover_y, defender, player, synthetic
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (game_id, event_number) DO NOTHING;
    """, rows)

    cur.execute("SELECT event_number, event_id FROM events WHERE game_id = %s", (game_id,))
    eid_map = {en: eid for en, eid in cur.fetchall()}

    line_rows = []
    for idx, ev in enumerate(events, 1):
        etype = ev.get("type")
        if etype not in LINE_EVENT_TYPES or idx not in eid_map: continue
        line_type = "D" if etype == 1 else "O" if etype == 2 else None
        for pos, pid in enumerate(ev.get("line", []), 1):
            line_rows.append((eid_map[idx], game_id, pid, line_type, pos, ev.get("team")))
    if line_rows:
        execute_batch(cur, """
            INSERT INTO line_players (event_id, game_id, player_id, line_type, position_order, team)
            VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (event_id, player_id) DO NOTHING;
        """, line_rows)

    conn.commit(); cur.close(); conn.close()

# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    import sys
    reset = '--reset' in sys.argv  # only wipe + recreate schema if explicitly requested

    on_railway = bool(os.environ.get('DATABASE_URL'))
    print(f"\nTarget: {'Railway' if on_railway else 'local'}")

    if reset:
        if on_railway:
            print("ERROR: --reset not allowed on Railway. Aborting.")
            sys.exit(1)
        print("\n[1/5] Creating database...")
        create_database()
        print("\n[2/5] Creating schema...")
        create_schema()
        print("  Schema created.")
    else:
        print("\nRunning in incremental mode (SKIP_EXISTING=True, schema untouched).")

    years_str = ",".join(str(y) for y in YEARS)

    print(f"\n[3/5] Fetching teams and players ({years_str})...")
    teams = get_teams_data(years=years_str)
    insert_teams(teams)
    print(f"  {len(teams)} team records inserted.")
    players = get_players_data(years=years_str)
    insert_players(players)
    print(f"  {len(players)} player records inserted.")

    print(f"\n[4/5] Fetching game list...")
    all_games = []
    for year in YEARS:
        games = get_games(date=f"{year}-01:{year}-12")
        print(f"  {year}: {len(games)} games")
        all_games.extend(games)
    print(f"  Total: {len(all_games)} games")

    print(f"\n[5/5] Inserting game events...")
    stats  = {"processed": 0, "skipped": 0, "errors": 0, "total_events": 0, "synthetic": 0}
    errors = []

    for game in tqdm(all_games, unit="game"):
        gid = game["game_id"]
        try:
            if SKIP_EXISTING:
                conn = get_connection(); cur = conn.cursor()
                cur.execute("SELECT 1 FROM games WHERE game_id = %s", (gid,))
                exists = cur.fetchone() is not None
                cur.close(); conn.close()
                if exists:
                    stats["skipped"] += 1
                    continue

            cleaned = clean_data([gid])
            if not cleaned or gid not in cleaned:
                stats["errors"] += 1
                errors.append({"game_id": gid, "error": "No event data returned"})
                continue

            evs = cleaned[gid]
            evs = insert_missing_turnovers(evs, seed=42)
            evs = normalize_coordinates(evs, game["away_team_id"])
            evs = average_turnover_coordinates(evs)
            insert_game_data(gid, evs, game["home_team_id"], game["away_team_id"],
                             game["home_score"], game["away_score"], game["year"])

            stats["processed"]    += 1
            stats["total_events"] += len(evs)
            stats["synthetic"]    += sum(1 for e in evs if e.get("synthetic"))
            time.sleep(API_SLEEP_S)

        except Exception as exc:
            stats["errors"] += 1
            errors.append({"game_id": gid, "error": str(exc)})

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)
    print(f"  Processed : {stats['processed']}")
    print(f"  Skipped   : {stats['skipped']}  (already in DB)")
    print(f"  Errors    : {stats['errors']}")
    print(f"  Events    : {stats['total_events']:,}")
    print(f"  Synthetic : {stats['synthetic']}")
    if errors:
        print(f"\n  First {min(5, len(errors))} errors:")
        for e in errors[:5]:
            print(f"    {e['game_id']}: {e['error']}")

    # ── Verify ────────────────────────────────────────────────────────────────
    conn = get_connection(); cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM games;");       print(f"\n  Games       : {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM events;");      print(f"  Events      : {cur.fetchone()[0]:,}")
    cur.execute("SELECT COUNT(*) FROM line_players;");print(f"  Line players: {cur.fetchone()[0]:,}")
    cur.execute("SELECT year, COUNT(*) FROM games GROUP BY year ORDER BY year;")
    print("\n  Games by year:")
    for row in cur.fetchall():
        print(f"    {row[0]}: {row[1]}")
    cur.close(); conn.close()
    print("\n✅ Database ready.\n")


if __name__ == "__main__":
    main()
