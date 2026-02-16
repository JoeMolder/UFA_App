#!/usr/bin/env python3
"""
Complete data ingestion pipeline with coordinate averaging.

This script combines all functions from:
- data_cleaning_functions.ipynb
- postgres_setup.ipynb

And runs the full pipeline to populate the database with cleaned,
coordinate-averaged data.
"""

import requests
import psycopg2
from psycopg2.extras import execute_batch, RealDictCursor
from typing import List, Dict, Optional, Tuple
import random
import time
from tqdm import tqdm

# ============================================================================
# Database Configuration
# ============================================================================

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

# ============================================================================
# API Functions (from data_cleaning_functions.ipynb)
# ============================================================================

def get_teams_data(years="all"):
    """Fetches team data from the UFA Stats API."""
    url = "https://www.backend.ufastats.com/api/v1/teams"
    params = {'years': years}

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json().get("data", [])

        teams = []
        for team in data:
            teams.append({
                'team_id': team.get('teamID'),
                'team_name': team.get('fullName'),
                'division': team.get('division', {}).get('name'),
                'year': team.get('year')
            })

        return teams

    except requests.exceptions.RequestException as e:
        print(f"Error fetching teams data: {e}")
        return []

def get_players_data(years="all", team_ids=None):
    """Fetches player data from the UFA Stats API."""
    url = "https://www.backend.ufastats.com/api/v1/players"

    params = {'years': years}
    if team_ids:
        params['teamIDs'] = team_ids

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json().get("data", [])

        players = []
        for player in data:
            player_id = player.get('playerID')
            first_name = player.get('firstName', '')
            last_name = player.get('lastName', '')
            full_name = f"{first_name} {last_name}".strip()

            teams = player.get('teams', [])
            if teams:
                for team in teams:
                    players.append({
                        'player_id': player_id,
                        'full_name': full_name,
                        'team_id': team.get('teamID'),
                        'year': team.get('year'),
                        'active': team.get('active', True),
                        'jersey_number': team.get('jerseyNumber')
                    })
            else:
                players.append({
                    'player_id': player_id,
                    'full_name': full_name,
                    'team_id': None,
                    'year': None,
                    'active': None,
                    'jersey_number': None
                })

        return players

    except requests.exceptions.RequestException as e:
        print(f"Error fetching players data: {e}")
        return []

def get_games(date=None, game_ids=None, statuses="Final", team_ids=None):
    """Fetches games from the UFA Stats API."""
    url = "https://www.backend.ufastats.com/api/v1/games"

    params = {}
    if date:
        params['date'] = date
    elif game_ids:
        params['gameIDs'] = game_ids
    else:
        raise ValueError("Must provide either 'date' or 'game_ids' parameter")

    if statuses:
        params['statuses'] = statuses
    if team_ids:
        params['teamIDs'] = team_ids

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json().get("data", [])

        games = []
        for game in data:
            game_id = game.get('gameID')
            year = int(game_id[:4]) if game_id else None
            game_date = game_id[:10] if game_id else None

            games.append({
                'game_id': game_id,
                'home_team_id': game.get('homeTeamID'),
                'away_team_id': game.get('awayTeamID'),
                'home_score': game.get('homeScore'),
                'away_score': game.get('awayScore'),
                'year': year,
                'game_date': game_date,
                'status': game.get('status')
            })

        return games

    except requests.exceptions.RequestException as e:
        print(f"Error fetching games data: {e}")
        return []

# Continue in next message...
