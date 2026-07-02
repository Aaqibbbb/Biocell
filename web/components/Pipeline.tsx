import { SectionHeading } from "./SectionHeading";

const STAGES = [
  { n: "01", title: "Whole slide", note: "Gigapixel WSI / mask" },
  { n: "02", title: "Cell detection", note: "Segmentation instances" },
  { n: "03", title: "Cell typing", note: "Morphology + FM embeddings" },
  { n: "04", title: "Biological graph", note: "KNN · Delaunay · contact" },
  { n: "05", title: "Graph foundation model", note: "GCN → Graph Transformer" },
  { n: "06", title: "Reasoning", note: "Subtype · survival · TME" },
  { n: "07", title: "Discovery", note: "Biomarkers · interactions" },
];

export function Pipeline() {
  return (
    <section id="pipeline" className="section">
      <div className="container-x">
        <SectionHeading
          eyebrow="The pipeline"
          title="From pixels to biological reasoning"
          description="Every stage is a swappable module. Instead of image → CNN → prediction, CellGraphFM reasons over cells and their spatial relationships the way a pathologist does."
        />

        <ol className="mt-12 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-7">
          {STAGES.map((s) => (
            <li key={s.n} className="glass glass-hover p-4">
              <span className="stat-num text-xs font-medium text-accent">{s.n}</span>
              <h3 className="mt-2 text-sm font-semibold leading-tight">{s.title}</h3>
              <p className="mt-1 text-xs leading-snug text-ink-muted">{s.note}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
