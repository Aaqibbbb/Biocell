import { SectionHeading } from "./SectionHeading";

const DIMS = [
  { n: 1, t: "Cellular representation", d: "Morphology + foundation-model embeddings per cell." },
  { n: 2, t: "Graph construction", d: "KNN, radius, Delaunay, and contact graphs." },
  { n: 3, t: "Multi-scale graphs", d: "Cell → microenvironment → region → slide." },
  { n: 4, t: "Neural architectures", d: "GCN, SAGE, GAT, GATv2, GIN, Graph Transformer." },
  { n: 5, t: "Self-supervised learning", d: "Masking + GraphMAE — GPT for cell graphs." },
  { n: 6, t: "Biological reasoning", d: "Subtype, survival, mutation, immune phenotype." },
  { n: 7, t: "Explainability", d: "Cell-type interactions, not opaque heatmaps." },
  { n: 8, t: "Foundation models", d: "UNI, Virchow, Prov-GigaPath, CONCH interfaces." },
  { n: 9, t: "Multimodal learning", d: "Image graph + transcriptomics + clinical text." },
  { n: 10, t: "Scientific discovery", d: "Novel biomarkers and microenvironments." },
];

export function Dimensions() {
  return (
    <section id="research" className="section">
      <div className="container-x">
        <SectionHeading
          eyebrow="Research"
          title="Ten dimensions, one framework"
          description="The project is organized around ten research dimensions — each mapped to a concrete, swappable module in the codebase."
        />

        <div className="mt-12 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
          {DIMS.map((d) => (
            <div key={d.n} className="glass glass-hover p-4">
              <span className="stat-num text-xs font-semibold text-accent-3">
                {String(d.n).padStart(2, "0")}
              </span>
              <h3 className="mt-2 text-sm font-semibold leading-tight">{d.t}</h3>
              <p className="mt-1.5 text-xs leading-snug text-ink-muted">{d.d}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
