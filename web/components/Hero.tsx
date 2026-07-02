const REPO = "https://github.com/Aaqibbbb/Biocell";

const STATS = [
  { value: "6", label: "GNN architectures" },
  { value: "8", label: "Foundation-model interfaces" },
  { value: "10", label: "Research dimensions" },
];

export function Hero() {
  return (
    <section className="relative overflow-hidden">
      <div className="grid-overlay pointer-events-none absolute inset-0" aria-hidden />
      <div className="container-x relative pb-16 pt-20 sm:pt-28">
        <div className="mx-auto max-w-3xl text-center">
          <span className="chip mx-auto">
            <span className="h-1.5 w-1.5 rounded-full bg-accent-2" />
            Open-source graph foundation model
          </span>

          <h1 className="mt-6 text-balance text-4xl font-semibold leading-[1.05] tracking-tight sm:text-6xl">
            Tissue is not pixels.
            <br />
            <span className="text-gradient">It&apos;s a graph of living cells.</span>
          </h1>

          <p className="mx-auto mt-6 max-w-2xl text-balance text-lg leading-relaxed text-ink-secondary">
            CellGraphFM represents a whole slide as interacting biological
            entities — cells, neighbourhoods, and tissue architecture — and learns
            universal representations that generalize across cancers, organs, and
            clinical tasks.
          </p>

          <div className="mt-9 flex flex-wrap items-center justify-center gap-3">
            <a
              href="#explorer"
              className="rounded-full bg-accent px-5 py-2.5 text-sm font-semibold text-white shadow-[0_8px_30px_-8px_rgba(57,135,229,0.7)] transition-transform hover:-translate-y-0.5"
            >
              Explore the graph
            </a>
            <a
              href={REPO}
              target="_blank"
              rel="noreferrer"
              className="rounded-full border border-line-strong bg-white/5 px-5 py-2.5 text-sm font-semibold text-ink transition-colors hover:bg-white/10"
            >
              View source
            </a>
          </div>

          <dl className="mx-auto mt-14 grid max-w-xl grid-cols-3 gap-4">
            {STATS.map((s) => (
              <div key={s.label} className="glass px-4 py-5">
                <dt className="stat-num text-3xl font-semibold text-ink">{s.value}</dt>
                <dd className="mt-1 text-xs leading-snug text-ink-muted">{s.label}</dd>
              </div>
            ))}
          </dl>
        </div>
      </div>
    </section>
  );
}
