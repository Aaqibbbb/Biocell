/**
 * Palette — the dataviz skill's validated categorical ramp (dark-mode steps),
 * assigned to cell types in the skill's fixed CVD-safe slot order. Colours
 * carry identity; text always wears ink tokens, never a series colour.
 */

export type CellType = {
  id: number;
  name: string;
  short: string;
  color: string;
};

/** Fixed slot order (blue, aqua, yellow, green, violet, red) — CVD-safe. */
export const CELL_TYPES: CellType[] = [
  { id: 0, name: "Tumor cell", short: "Tumor", color: "#3987e5" },
  { id: 1, name: "T cell", short: "T cell", color: "#199e70" },
  { id: 2, name: "Fibroblast", short: "Fibroblast", color: "#c98500" },
  { id: 3, name: "Macrophage", short: "Macrophage", color: "#4c9f4c" },
  { id: 4, name: "Endothelial", short: "Endothelial", color: "#9085e9" },
  { id: 5, name: "B cell", short: "B cell", color: "#e66767" },
];

export const N_TYPES = CELL_TYPES.length;

export const colorFor = (type: number): string =>
  CELL_TYPES[((type % N_TYPES) + N_TYPES) % N_TYPES].color;

/** Sequential blue ramp (light -> dark) for magnitude encodings (heatmaps). */
export const SEQ_BLUE = [
  "#0d1b2f",
  "#123458",
  "#184f95",
  "#256abf",
  "#3987e5",
  "#6da7ec",
  "#9ec5f4",
];

/** Chart chrome / ink for the dark chart surface. */
export const INK = {
  primary: "#ffffff",
  secondary: "#c3c2b7",
  muted: "#898781",
  grid: "#2c2c2a",
  surface: "#1a1a19",
};

/** Map a value in [0,1] onto the sequential blue ramp. */
export function seqColor(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const idx = Math.round(clamped * (SEQ_BLUE.length - 1));
  return SEQ_BLUE[idx];
}
