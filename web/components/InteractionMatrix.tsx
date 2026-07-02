"use client";

import { useMemo, useState } from "react";
import { generateTissue, interactionMatrix } from "@/lib/tissue";
import { CELL_TYPES, INK, SEQ_BLUE, seqColor } from "@/lib/palette";

const N = CELL_TYPES.length;
const CELL = 46;
const GAP = 6;
const PAD_L = 96;
const PAD_T = 84;
const STEP = CELL + GAP;

export function InteractionMatrix() {
  const { matrix, max } = useMemo(() => {
    const graph = generateTissue({ nCells: 340, k: 6, seed: 7 });
    const m = interactionMatrix(graph);
    let mx = 1;
    for (const row of m) for (const v of row) mx = Math.max(mx, v);
    return { matrix: m, max: mx };
  }, []);

  const [hover, setHover] = useState<{ a: number; b: number } | null>(null);

  const width = PAD_L + N * STEP + 8;
  const height = PAD_T + N * STEP + 8;

  const readout = hover
    ? `${CELL_TYPES[hover.a].name} ↔ ${CELL_TYPES[hover.b].name} — ${matrix[hover.a][hover.b]} contacts`
    : "Hover a cell to read the contact count between two cell types.";

  return (
    <div className="glass p-5 sm:p-6">
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <h3 className="text-sm font-semibold tracking-tight">
          Cell-type interaction matrix
        </h3>
        <span className="text-xs text-ink-muted">edges between types</span>
      </div>

      <div className="overflow-x-auto">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="h-auto w-full min-w-[420px]"
          role="img"
          aria-label="Heatmap of contact counts between cell types"
        >
          {/* Column labels */}
          {CELL_TYPES.map((t, j) => (
            <text
              key={`c-${t.id}`}
              x={PAD_L + j * STEP + CELL / 2}
              y={PAD_T - 14}
              textAnchor="start"
              fontSize="11"
              fill={INK.muted}
              transform={`rotate(-45 ${PAD_L + j * STEP + CELL / 2} ${PAD_T - 14})`}
            >
              {t.short}
            </text>
          ))}
          {/* Row labels */}
          {CELL_TYPES.map((t, i) => (
            <text
              key={`r-${t.id}`}
              x={PAD_L - 12}
              y={PAD_T + i * STEP + CELL / 2 + 4}
              textAnchor="end"
              fontSize="11"
              fill={INK.muted}
            >
              {t.short}
            </text>
          ))}
          {/* Cells */}
          {matrix.map((row, i) =>
            row.map((v, j) => {
              const isHover = hover?.a === i && hover?.b === j;
              return (
                <rect
                  key={`${i}-${j}`}
                  x={PAD_L + j * STEP}
                  y={PAD_T + i * STEP}
                  width={CELL}
                  height={CELL}
                  rx={5}
                  fill={seqColor(v / max)}
                  stroke={isHover ? INK.primary : "transparent"}
                  strokeWidth={isHover ? 1.5 : 0}
                  onMouseEnter={() => setHover({ a: i, b: j })}
                  onMouseLeave={() => setHover(null)}
                  style={{ cursor: "pointer" }}
                />
              );
            }),
          )}
        </svg>
      </div>

      {/* Readout */}
      <p className="mt-4 min-h-5 text-sm text-ink-secondary">{readout}</p>

      {/* Colorbar legend */}
      <div className="mt-3 flex items-center gap-3">
        <span className="text-xs text-ink-muted">0</span>
        <div
          className="h-2 flex-1 rounded-full"
          style={{
            background: `linear-gradient(to right, ${SEQ_BLUE.join(",")})`,
          }}
        />
        <span className="stat-num text-xs text-ink-muted">{max}</span>
      </div>
    </div>
  );
}
