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

export interface PredictionResponse {
  grid: number[][];
  extent: [number, number, number, number];
}

export interface PlayerStats {
  total_throws: number;
  completion_pct: number;
  goal_pct: number;
  avg_throw_dist: number;
  avg_throw_depth: number;
  huck_rate: number;
  avg_lateral_dist: number;
  avg_dist_from_center: number;
}

export interface ClusterSummary {
  count: number;
  avg_completion_pct: number;
  avg_throw_dist: number;
  avg_throw_depth: number;
  avg_huck_rate: number;
  avg_goal_pct: number;
  avg_total_throws: number;
  avg_lateral_dist: number;
  avg_dist_from_center: number;
}

export interface EmbeddingsResponse {
  players: string[];
  coordinates: number[][];
  clusters: number[];
  player_stats: Record<string, PlayerStats>;
  cluster_summaries: Record<string, ClusterSummary>;
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

  // Get player list for predictions
  getPlayers: async (): Promise<string[]> => {
    const response = await apiClient.get<string[]>('/players');
    return response.data;
  },

  // Get UMAP player embeddings
  getEmbeddings: async (): Promise<EmbeddingsResponse> => {
    const response = await apiClient.get<EmbeddingsResponse>('/embeddings/players');
    return response.data;
  },

  // Get throw prediction heatmap
  predictThrows: async (
    player: string,
    x: number,
    y: number,
    gridSize = 30
  ): Promise<PredictionResponse> => {
    const response = await apiClient.get<PredictionResponse>('/predict/throws', {
      params: { player, x, y, grid_size: gridSize },
    });
    return response.data;
  },
};
