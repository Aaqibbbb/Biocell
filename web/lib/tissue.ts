/**
 * Client-side synthetic tissue graph — a faithful, dependency-free port of the
 * ideas in the Python `cellgraphfm` core: spatially-clustered cells with a
 * k-nearest-neighbour graph over 3D coordinates. Deterministic (seeded) so the
 * artifact is reproducible.
 */

import { N_TYPES } from "./palette";

export type Cell = {
  position: [number, number, number];
  type: number;
};

export type TissueGraph = {
  cells: Cell[];
  edges: Array<[number, number]>;
};

export type TissueParams = {
  nCells?: number;
  nTypes?: number;
  size?: number;
  k?: number;
  seed?: number;
};

/** Small, fast, seedable PRNG (mulberry32). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Sum of three uniforms → approximately Gaussian, centred, in ~[-1.5, 1.5]. */
function gaussish(rng: () => number): number {
  return rng() + rng() + rng() - 1.5;
}

export function generateTissue(params: TissueParams = {}): TissueGraph {
  const {
    nCells = 300,
    nTypes = N_TYPES,
    size = 10,
    k = 6,
    seed = 7,
  } = params;

  const rng = mulberry32(seed);
  const half = size / 2;

  // Spatial cluster centres per cell type → same-type cells sit together,
  // producing meaningful neighbourhood enrichment.
  const centers: Array<[number, number, number]> = Array.from(
    { length: nTypes },
    () => [
      (rng() - 0.5) * size * 0.85,
      (rng() - 0.5) * size * 0.85,
      (rng() - 0.5) * size * 0.85,
    ],
  );

  const spread = size * 0.16;
  const cells: Cell[] = [];
  for (let i = 0; i < nCells; i++) {
    const type = Math.floor(rng() * nTypes) % nTypes;
    const c = centers[type];
    const clamp = (v: number) => Math.max(-half, Math.min(half, v));
    cells.push({
      type,
      position: [
        clamp(c[0] + gaussish(rng) * spread * 2),
        clamp(c[1] + gaussish(rng) * spread * 2),
        clamp(c[2] + gaussish(rng) * spread * 2),
      ],
    });
  }

  const edges = knnEdges(cells, k);
  return { cells, edges };
}

function knnEdges(cells: Cell[], k: number): Array<[number, number]> {
  const n = cells.length;
  const seen = new Set<number>();
  const edges: Array<[number, number]> = [];

  for (let i = 0; i < n; i++) {
    const dists: Array<[number, number]> = [];
    const [xi, yi, zi] = cells[i].position;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const [xj, yj, zj] = cells[j].position;
      const dx = xi - xj;
      const dy = yi - yj;
      const dz = zi - zj;
      dists.push([dx * dx + dy * dy + dz * dz, j]);
    }
    dists.sort((p, q) => p[0] - q[0]);
    for (let m = 0; m < Math.min(k, dists.length); m++) {
      const j = dists[m][1];
      const a = Math.min(i, j);
      const b = Math.max(i, j);
      const key = a * n + b;
      if (!seen.has(key)) {
        seen.add(key);
        edges.push([a, b]);
      }
    }
  }
  return edges;
}

/**
 * Cell-type interaction matrix: counts[a][b] is the number of undirected edges
 * connecting a type-a cell to a type-b cell (symmetric).
 */
export function interactionMatrix(graph: TissueGraph, nTypes = N_TYPES): number[][] {
  const m: number[][] = Array.from({ length: nTypes }, () =>
    new Array(nTypes).fill(0),
  );
  for (const [i, j] of graph.edges) {
    const a = graph.cells[i].type;
    const b = graph.cells[j].type;
    m[a][b] += 1;
    if (a !== b) m[b][a] += 1;
  }
  return m;
}

/** Per-type cell counts. */
export function typeCounts(graph: TissueGraph, nTypes = N_TYPES): number[] {
  const counts = new Array(nTypes).fill(0);
  for (const c of graph.cells) counts[c.type] += 1;
  return counts;
}
