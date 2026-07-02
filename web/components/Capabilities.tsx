import { SectionHeading } from "./SectionHeading";

const CAPS = [
  {
    title: "Tissue & ROI detection",
    module: "wsi.tissue",
    desc: "Otsu saturation thresholding with connected-component analysis locates tissue and regions of interest on a slide thumbnail.",
  },
  {
    title: "Composite Imaging (EDoF)",
    module: "wsi.focus",
    desc: "Z-stack focus stacking fuses multiple focal planes into a single all-in-focus image via per-pixel Laplacian energy.",
  },
  {
    title: "WSI tiling + embeddings",
    module: "wsi.tiling",
    desc: "Tile gigapixel slides, filter by tissue fraction, and embed each tile with pathology foundation models (UNI, Virchow…).",
  },
  {
    title: "IHC biomarker quantification",
    module: "biomarkers.ihc",
    desc: "Ruifrok–Johnston colour deconvolution with Ki-67 / HER2 / ER / PR positivity index and H-score.",
  },
  {
    title: "Scan quality control",
    module: "qc",
    desc: "Sharpness, contrast and blur metrics flag focus and staining problems before downstream analysis runs.",
  },
  {
    title: "Tumor–immune scoring",
    module: "scoring",
    desc: "TIL density, immune infiltration, tumour cellularity, and spatial mixing computed directly from the cell graph.",
  },
  {
    title: "Explainable interactions",
    module: "explain",
    desc: "Interaction matrices, permutation neighbourhood enrichment, and interaction-path mining — in cell-type language.",
  },
  {
    title: "Self-supervised pretraining",
    module: "ssl",
    desc: "GraphMAE-style node masking with a scaled-cosine reconstruction loss learns representations without labels.",
  },
];

export function Capabilities() {
  return (
    <section id="capabilities" className="section">
      <div className="container-x">
        <SectionHeading
          eyebrow="Capabilities"
          title="A full computational-pathology toolkit"
          description="Software analogs of modern digital-pathology workflows — tissue detection, extended-depth-of-field imaging, biomarker quantification, quality control and scoring — all implemented in a dependency-light, tested Python core."
        />

        <div className="mt-12 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {CAPS.map((c) => (
            <div key={c.module} className="glass glass-hover flex flex-col p-5">
              <div className="mb-3 h-1 w-8 rounded-full bg-accent/70" />
              <h3 className="text-sm font-semibold leading-tight">{c.title}</h3>
              <p className="mt-2 flex-1 text-[0.82rem] leading-relaxed text-ink-secondary">
                {c.desc}
              </p>
              <code className="mt-4 w-fit rounded-md bg-white/[0.04] px-2 py-1 font-mono text-[0.7rem] text-ink-muted">
                cellgraphfm.{c.module}
              </code>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
