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

interface BatchPredictionResponseRaw {
  grids: Record<string, string>;
  x_positions: number[];
  y_positions: number[];
  extent: [number, number, number, number];
}

function float16ToFloat32(h: number): number {
  const s = (h & 0x8000) >> 15
  const e = (h & 0x7c00) >> 10
  const f = h & 0x03ff
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024)
  if (e === 31) return f ? NaN : (s ? -Infinity : Infinity)
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024)
}

function decodeFloat16Grid(b64: string, rows: number, cols: number): number[][] {
  const binary = atob(b64)
  const result: number[][] = new Array(rows)
  for (let r = 0; r < rows; r++) {
    result[r] = new Array(cols)
    for (let c = 0; c < cols; c++) {
      const i = r * cols + c
      const lo = binary.charCodeAt(i * 2)
      const hi = binary.charCodeAt(i * 2 + 1)
      result[r][c] = float16ToFloat32(lo | (hi << 8))
    }
  }
  return result
}

function decodeBatchResponse(raw: BatchPredictionResponseRaw, rows: number, cols: number): BatchPredictionResponse {
  const grids: Record<string, number[][]> = {}
  for (const [key, b64] of Object.entries(raw.grids)) {
    grids[key] = decodeFloat16Grid(b64, rows, cols)
  }
  return { ...raw, grids }
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
  sample_size: number | null;
  pull_landing: { x: number; y: number };
  scoring_rate: number | null;
  mode?: string;
}

export interface PullPlaySampleResponse {
  sequences: PullPlayThrow[][];
  pull_landing: { x: number; y: number };
  n_samples: number;
  team: string | null;
}

export interface PullPlayCluster {
  cluster_id: number;
  throws: PullPlayThrow[];
  count: number;
  frequency: number;
}

export interface PullPlayClustersResponse {
  clusters: PullPlayCluster[];
  n_clusters: number;
}

export interface PullPlayHotspot {
  x: number;
  y: number;
  count: number;
  relative_freq: number;
}

export interface ZoneThrow {
  from_x: number; from_y: number;
  to_x: number;   to_y: number;
}

export interface FieldZone {
  zone_id: number;
  col: number;
  row: number;
  x_range: [number, number];
  y_range: [number, number];
  count: number;
  relative_density: number;
  throws: ZoneThrow[];
}

export interface ZonePatternsResponse {
  zones: FieldZone[];
  zone_cols: number;
  zone_rows: number;
  total: number;
}

export interface EmbeddingsResponse {
  players: string[];
  coordinates: number[][];
  clusters: number[];
  player_stats: Record<string, PlayerStats>;
  cluster_summaries: Record<string, ClusterSummary>;
  name_map: Record<string, string>;
}

export interface EPVResponse {
  grid: number[][];
  extent: [number, number, number, number];
  throw_idx: number;
}

export interface CompletionHeatmapResponse {
  grid: number[][];
  extent: [number, number, number, number];
}

export interface PlayerOption {
  id: string;
  name: string;
}

export interface ThrowExchange {
  count: number;
  completion_pct: number;
}

export interface SynergyPairResponse {
  player1: PlayerOption;
  player2: PlayerOption;
  shared_possessions: number;
  combined_scoring_rate: number;
  p1_scoring_rate: number;
  p2_scoring_rate: number;
  synergy_delta: number;
  p1_to_p2: ThrowExchange;
  p2_to_p1: ThrowExchange;
}

export interface LineupPredictResponse {
  probability: number;
  players: PlayerOption[];
  known_players: number;
}

export interface TeamSynergyPair {
  player1: { id: string; name: string };
  player2: { id: string; name: string };
  shared_possessions: number;
  combined_rate: number;
  p1_rate: number;
  p2_rate: number;
  synergy_delta: number;
}

export interface TeamPlayer {
  id: string;
  name: string;
  hold_rate: number;
  possessions: number;
}

export interface RosterPlayer {
  id: string;
  name: string;
  o_appearances: number;
  d_appearances: number;
}

export interface TeamResponse {
  team_id: string;
  team_name: string;
  division: string;
  available_years: number[];
  selected_year: number | null;
  record: { wins: number; losses: number; games: number };
  o_line_rate: number;
  top_players: TeamPlayer[];
  roster: RosterPlayer[];
  top_synergies: TeamSynergyPair[];
}

export interface CompletionPredictResponse {
  probability: number;
  thrower: string;
}

export interface PlayerSeason {
  year: number;
  team: string;
  o_possessions: number;
  d_possessions: number;
  throw_attempts: number;
  completions: number;
  completion_pct: number;
  assists: number;
  goals: number;
  turnovers: number;
  huck_attempts: number;
  huck_completions: number;
  huck_pct: number;
  avg_throw_dist: number;
  avg_throw_depth: number;
  catches: number;
  blocks: number;
  o_hold_rate: number;
}

export interface PlayerConnection {
  id: string;
  name: string;
  count: number;
  completion_pct: number;
}

export interface PlayerSynergyPartner {
  id: string;
  name: string;
  shared_possessions: number;
  combined_rate: number;
  synergy_delta: number;
}

export interface PlayerResponse {
  player: PlayerOption;
  available_years: number[];
  seasons: PlayerSeason[];
  career: PlayerSeason;
  top_targets: PlayerConnection[];
  top_throwers: PlayerConnection[];
  synergy_partners: PlayerSynergyPartner[];
}

export interface ThrowTendencyBin {
  angle_deg: number;
  count: number;
  pct: number;
  avg_dist: number;
}

export interface ThrowTendencies {
  bins: ThrowTendencyBin[];
  total_throws: number;
  max_avg_dist: number;
}

export interface BlockTypes {
  huck: number;
  short: number;
  reset: number;
  total: number;
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
  getPlayers: async (): Promise<PlayerOption[]> => {
    const response = await apiClient.get<PlayerOption[]>('/players');
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

  getTurnoversBatch: async (): Promise<BatchPredictionResponse> => {
    const response = await apiClient.get<BatchPredictionResponseRaw>('/predict/turnovers/batch');
    return decodeBatchResponse(response.data, 120, 100);
  },

  getBlocksBatch: async (): Promise<BatchPredictionResponse> => {
    const response = await apiClient.get<BatchPredictionResponseRaw>('/predict/blocks/batch');
    return decodeBatchResponse(response.data, 120, 100);
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

  // Get pull play sequence (CVAE model or data average)
  getPullPlaySequence: async (
    pullX: number,
    pullY: number,
    team?: string,
    mode: 'model' | 'average' = 'model',
    radius = 15
  ): Promise<PullPlayResponse> => {
    const params: Record<string, string | number> = { pull_x: pullX, pull_y: pullY, radius, mode };
    if (team) params.team = team;
    const response = await apiClient.get<PullPlayResponse>('/pull-play/sequence', { params });
    return response.data;
  },

  // Sample multiple play sequences from the CVAE latent space
  samplePullPlays: async (
    pullX: number,
    pullY: number,
    team?: string,
    nSamples = 5
  ): Promise<PullPlaySampleResponse> => {
    const params: Record<string, string | number> = { pull_x: pullX, pull_y: pullY, n_samples: nSamples };
    if (team) params.team = team;
    const response = await apiClient.get<PullPlaySampleResponse>('/pull-play/sample', { params });
    return response.data;
  },

  // Get common play archetypes decoded conditioned on a specific pull landing
  getPullPlayClusters: async (
    pullX: number,
    pullY: number,
    team?: string
  ): Promise<PullPlayClustersResponse> => {
    const params: Record<string, string | number> = { pull_x: pullX, pull_y: pullY };
    if (team) params.team = team;
    const response = await apiClient.get<PullPlayClustersResponse>('/pull-play/clusters', { params });
    return response.data;
  },

  // Get zone-level possession start patterns (pulls + turnovers)
  getZonePatterns: async (team?: string, zoneCols = 4, zoneRows = 3): Promise<ZonePatternsResponse> => {
    const params: Record<string, string | number> = { zone_cols: zoneCols, zone_rows: zoneRows };
    if (team) params.team = team;
    const response = await apiClient.get<ZonePatternsResponse>('/possession/zone-patterns', { params });
    return response.data;
  },

  // Get most common pull landing positions (hotspots) for a team or all teams
  getPullPlayHotspots: async (team?: string): Promise<PullPlayHotspot[]> => {
    const params: Record<string, string> = {};
    if (team) params.team = team;
    const response = await apiClient.get<{ hotspots: PullPlayHotspot[] }>('/pull-play/hotspots', { params });
    return response.data.hotspots;
  },

  // Get EPV (Expected Possession Value) heatmap
  getEPVHeatmap: async (
    throwIdx: number,
    team?: string,
    model: 'xgb' | 'nn' = 'xgb',
    quarter?: number
  ): Promise<EPVResponse> => {
    const params: Record<string, string | number> = { throw_idx: throwIdx, model };
    if (team) params.team = team;
    if (quarter) params.quarter = quarter;
    const response = await apiClient.get<EPVResponse>('/epv/heatmap', { params });
    return response.data;
  },

  // Get list of throwers in the completion model
  getCompletionThrowers: async (): Promise<PlayerOption[]> => {
    const response = await apiClient.get<PlayerOption[]>('/completion/throwers');
    return response.data;
  },

  // Get completion probability heatmap from a given origin
  getCompletionHeatmap: async (
    thrower: string,
    fromX: number,
    fromY: number
  ): Promise<CompletionHeatmapResponse> => {
    const response = await apiClient.get<CompletionHeatmapResponse>('/completion/heatmap', {
      params: { thrower, from_x: fromX, from_y: fromY },
    });
    return response.data;
  },

  // Get completion probability for a single throw
  getCompletionPredict: async (
    thrower: string,
    fromX: number,
    fromY: number,
    toX: number,
    toY: number
  ): Promise<CompletionPredictResponse> => {
    const response = await apiClient.get<CompletionPredictResponse>('/completion/predict', {
      params: { thrower, from_x: fromX, from_y: fromY, to_x: toX, to_y: toY },
    });
    return response.data;
  },

  // Get batch throw predictions for all grid cells
  predictThrowsBatch: async (
    player: string,
    gridCellsX = 10,
    gridCellsY = 12,
    heatmapResolution = 30
  ): Promise<BatchPredictionResponse> => {
    const response = await apiClient.get<BatchPredictionResponseRaw>('/predict/throws/batch', {
      params: { player, grid_cells_x: gridCellsX, grid_cells_y: gridCellsY, heatmap_resolution: heatmapResolution },
    });
    return decodeBatchResponse(response.data, 120, 100);
  },

  // Line synergy: all players on O-lines
  getSynergyPlayers: async (): Promise<PlayerOption[]> => {
    const response = await apiClient.get<PlayerOption[]>('/synergy/players');
    return response.data;
  },

  // Line synergy: pair metrics
  getSynergyPair: async (player1: string, player2: string): Promise<SynergyPairResponse> => {
    const response = await apiClient.get<SynergyPairResponse>('/synergy/pair', {
      params: { player1, player2 },
    });
    return response.data;
  },

  // Lineup scoring predictor
  getLineupPredict: async (players: string[]): Promise<LineupPredictResponse> => {
    const [p1, p2, p3, p4, p5, p6, p7] = players;
    const response = await apiClient.get<LineupPredictResponse>('/lineup/predict', {
      params: { p1, p2, p3, p4, p5, p6, p7 },
    });
    return response.data;
  },

  // Get team page data
  getTeam: async (teamId: string, year?: number): Promise<TeamResponse> => {
    const params: Record<string, number> = {};
    if (year) params.year = year;
    const response = await apiClient.get<TeamResponse>(`/team/${teamId}`, { params });
    return response.data;
  },

  // Get player page data
  getPlayer: async (playerId: string, year?: number): Promise<PlayerResponse> => {
    const params: Record<string, number> = {};
    if (year) params.year = year;
    const response = await apiClient.get<PlayerResponse>(`/player/${playerId}`, { params });
    return response.data;
  },

  // Get throw direction tendencies for a player
  getPlayerThrowTendencies: async (playerId: string, year?: number): Promise<ThrowTendencies> => {
    const params: Record<string, number> = {};
    if (year) params.year = year;
    const response = await apiClient.get<ThrowTendencies>(`/player/${playerId}/throw-tendencies`, { params });
    return response.data;
  },

  getPlayerBlockTypes: async (playerId: string, year?: number): Promise<BlockTypes> => {
    const params: Record<string, number> = {};
    if (year) params.year = year;
    const response = await apiClient.get<BlockTypes>(`/player/${playerId}/block-types`, { params });
    return response.data;
  },
};
