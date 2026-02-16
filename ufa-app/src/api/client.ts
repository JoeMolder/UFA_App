import axios from 'axios';

// API base URL - FastAPI backend
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Types
export interface Game {
  game_id: string;
  home_team_id: string;
  away_team_id: string;
  home_score: number;
  away_score: number;
  game_date: string;
  year: number;
}

export interface GameEvent {
  event_id: number;
  event_number: number;
  event_type: number;
  team: string;
  thrower?: string;
  thrower_x?: number;
  thrower_y?: number;
  receiver?: string;
  receiver_x?: number;
  receiver_y?: number;
  turnover_x?: number;
  turnover_y?: number;
  defender?: string;
  synthetic?: boolean;
}

export interface StatsResponse {
  total_games: number;
  total_events: number;
  synthetic_events: number;
}

// API Functions
export const api = {
  // Get list of games
  getGames: async (limit = 10): Promise<Game[]> => {
    const response = await apiClient.get<Game[]>(`/games?limit=${limit}`);
    return response.data;
  },

  // Get single game details
  getGame: async (gameId: string): Promise<Game> => {
    const response = await apiClient.get<Game>(`/games/${gameId}`);
    return response.data;
  },

  // Get game events
  getGameEvents: async (gameId: string): Promise<GameEvent[]> => {
    const response = await apiClient.get<GameEvent[]>(`/games/${gameId}/events`);
    return response.data;
  },

  // Get stats summary
  getStats: async (): Promise<StatsResponse> => {
    const response = await apiClient.get<StatsResponse>('/stats/summary');
    return response.data;
  },
};
