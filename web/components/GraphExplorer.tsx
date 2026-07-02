"use client";

import { useMemo, useState } from "react";
import clsx from "clsx";
import { CellGraph3D } from "./CellGraph3D";
import { generateTissue, typeCounts } from "@/lib/tissue";
import { CELL_TYPES } from "@/lib/palette";

function Segmented<T extends number>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: readonly T[];
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-ink-muted">{label}</span>
      <div className="flex rounded-full border border-line bg-white/[0.03] p-0.5">
        {options.map((opt) => (
          <button
            key={opt}
            type="button"
            onClick={() => onChange(opt)}
            className={clsx(
              "rounded-full px-3 py-1 text-xs font-medium transition-colors",
              value === opt
                ? "bg-white/12 text-ink"
                : "text-ink-muted hover:text-ink-secondary",
            )}
          >
            {opt}
          </button>
        ))}
      </div>
    </div>
  );
}

export function GraphExplorer() {
  const [nCells, setNCells] = useState<number>(300);
  const [k, setK] = useState<number>(6);
  const [seed, setSeed] = useState<number>(7);
  const [rotate, setRotate] = useState<boolean>(true);

  const graph = useMemo(
    () => generateTissue({ nCells, k, seed }),
    [nCells, k, seed],
  );
  const counts = useMemo(() => typeCounts(graph), [graph]);
  const avgDegree = graph.cells.length
    ? (2 * graph.edges.length) / graph.cells.length
    : 0;

  return (
    <div className="glass overflow-hidden">
      {/* Controls */}
      <div className="flex flex-wrap items-center gap-x-6 gap-y-3 border-b border-line px-4 py-3">
        <Segmented label="Cells" value={nCells} options={[150, 300, 500] as const} onChange={setNCells} />
        <Segmented label="k-NN" value={k} options={[4, 6, 8] as const} onChange={setK} />
        <div className="ml-auto flex items-center gap-2">
          <button
            type="button"
            onClick={() => setRotate((r) => !r)}
            className="rounded-full border border-line bg-white/[0.03] px-3 py-1 text-xs font-medium text-ink-secondary transition-colors hover:text-ink"
          >
            {rotate ? "Pause" : "Rotate"}
          </button>
          <button
            type="button"
            onClick={() => setSeed((s) => (s * 1103515245 + 12345) & 0x7fffffff)}
            className="rounded-full border border-line-strong bg-white/[0.06] px-3 py-1 text-xs font-medium text-ink transition-colors hover:bg-white/[0.12]"
          >
            Regenerate
          </button>
        </div>
      </div>

      {/* 3D canvas */}
      <div className="relative h-[440px] sm:h-[560px]">
        <CellGraph3D graph={graph} autoRotate={rotate} />
        <div className="pointer-events-none absolute left-4 top-4 flex gap-2">
          <span className="chip stat-num">{graph.cells.length} cells</span>
          <span className="chip stat-num">{graph.edges.length} edges</span>
          <span className="chip stat-num">{avgDegree.toFixed(1)} avg degree</span>
        </div>
        <p className="pointer-events-none absolute bottom-3 right-4 text-xs text-ink-muted">
          drag to orbit · scroll to zoom
        </p>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-x-5 gap-y-2 border-t border-line px-4 py-3">
        {CELL_TYPES.map((t, i) => (
          <div key={t.id} className="flex items-center gap-2">
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: t.color }}
            />
            <span className="text-xs text-ink-secondary">{t.name}</span>
            <span className="stat-num text-xs text-ink-muted">{counts[i]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
