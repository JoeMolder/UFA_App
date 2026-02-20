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

export interface BatchPredictionResponse {
  grids: Record<string, number[][]>;
  x_positions: number[];
  y_positions: number[];
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

export interface TurnoverHeatmapResponse {
  throw_grid: number[][];
  turnover_grid: number[][];
  total_throws: number;
  total_turnovers: number;
  extent: [number, number, number, number];
}

export interface TurnoverOriginsResponse {
  grid: number[][];
  total_throws: number;
  total_turnovers: number;
  extent: [number, number, number, number];
}

export interface PullPlayThrow {
  from_x: number;
  from_y: number;
  to_x: number;
  to_y: number;
}

export interface PullPlayResponse {
  throws: PullPlayThrow[];
  sample_size: number;
  pull_landing: { x: number; y: number };
  scoring_rate: number;
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
  getGames: async (limit = 10, search = ''): Promise<Game[]> => {
    const params: Record<string, string | number> = { limit };
    if (search) params.search = search;
    const response = await apiClient.get<Game[]>('/games', { params });
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

  // Get team list
  getTeams: async (): Promise<string[]> => {
    const response = await apiClient.get<string[]>('/teams');
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

  // Get turnover prediction heatmap (flow model, no player)
  predictTurnovers: async (
    x: number,
    y: number,
    gridSize = 30
  ): Promise<PredictionResponse> => {
    const response = await apiClient.get<PredictionResponse>('/predict/turnovers', {
      params: { x, y, grid_size: gridSize },
    });
    return response.data;
  },

  // Get relative density (turnover/completion ratio) prediction
  predictRelativeDensity: async (
    x: number,
    y: number,
    gridSize = 30
  ): Promise<PredictionResponse> => {
    const response = await apiClient.get<PredictionResponse>('/predict/relative-density', {
      params: { x, y, grid_size: gridSize },
    });
    return response.data;
  },

  // Get turnover origins heatmap (where turnovers are thrown from)
  getTurnoverOrigins: async (
    player?: string,
    gridX = 50,
    gridY = 60,
    smooth = 2.0,
    team?: string,
    opponent?: string
  ): Promise<TurnoverOriginsResponse> => {
    const params: Record<string, string | number> = { grid_x: gridX, grid_y: gridY, smooth };
    if (player) params.player = player;
    if (team) params.team = team;
    if (opponent) params.opponent = opponent;
    const response = await apiClient.get<TurnoverOriginsResponse>('/heatmap/turnover-origins', { params });
    return response.data;
  },

  // Get turnover heatmap data
  getTurnoverHeatmap: async (
    gridX = 50,
    gridY = 60,
    smooth = 2.0
  ): Promise<TurnoverHeatmapResponse> => {
    const response = await apiClient.get<TurnoverHeatmapResponse>('/heatmap/turnovers', {
      params: { grid_x: gridX, grid_y: gridY, smooth },
    });
    return response.data;
  },

  // Get batch throwaway destination grids for all thrower positions
  getThrowawaysBatch: async (
    throwerGridX = 10,
    throwerGridY = 24,
    destGridX = 30,
    destGridY = 36,
    radius = 12
  ): Promise<BatchPredictionResponse> => {
    const response = await apiClient.get<BatchPredictionResponse>('/heatmap/throwaways/batch', {
      params: { thrower_grid_x: throwerGridX, thrower_grid_y: throwerGridY, dest_grid_x: destGridX, dest_grid_y: destGridY, radius },
    });
    return response.data;
  },

  // Get block prediction heatmap (flow model, no player)
  predictBlocks: async (
    x: number,
    y: number,
    gridSize = 30
  ): Promise<PredictionResponse> => {
    const response = await apiClient.get<PredictionResponse>('/predict/blocks', {
      params: { x, y, grid_size: gridSize },
    });
    return response.data;
  },

  // Get pull play sequence (expected throws after a pull)
  getPullPlaySequence: async (
    pullX: number,
    pullY: number,
    team?: string,
    radius = 15
  ): Promise<PullPlayResponse> => {
    const params: Record<string, string | number> = { pull_x: pullX, pull_y: pullY, radius };
    if (team) params.team = team;
    const response = await apiClient.get<PullPlayResponse>('/pull-play/sequence', { params });
    return response.data;
  },

  // Get batch throw predictions for all grid cells
  predictThrowsBatch: async (
    player: string,
    gridCellsX = 10,
    gridCellsY = 12,
    heatmapResolution = 30
  ): Promise<BatchPredictionResponse> => {
    const response = await apiClient.get<BatchPredictionResponse>('/predict/throws/batch', {
      params: { player, grid_cells_x: gridCellsX, grid_cells_y: gridCellsY, heatmap_resolution: heatmapResolution },
    });
    return response.data;
  },
};
