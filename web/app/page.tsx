import { Nav } from "@/components/Nav";
import { Hero } from "@/components/Hero";
import { GraphExplorer } from "@/components/GraphExplorer";
import { Pipeline } from "@/components/Pipeline";
import { Capabilities } from "@/components/Capabilities";
import { InteractionMatrix } from "@/components/InteractionMatrix";
import { Dimensions } from "@/components/Dimensions";
import { Platform } from "@/components/Platform";
import { Footer } from "@/components/Footer";
import { SectionHeading } from "@/components/SectionHeading";

const REPO = "https://github.com/Aaqibbbb/Biocell";

export default function Home() {
  return (
    <>
      <span id="top" />
      <Nav />
      <main className="flex-1">
        <Hero />

        {/* Live 3D artifact */}
        <section id="explorer" className="section pt-4">
          <div className="container-x">
            <SectionHeading
              align="center"
              eyebrow="Live artifact"
              title="Explore a tissue as a 3D cell graph"
              description="A deterministic synthetic section rendered client-side — the same graph construction the Python core performs. Orbit it, regenerate it, and watch cell-type neighbourhoods form."
            />
            <div className="mt-10">
              <GraphExplorer />
            </div>
          </div>
        </section>

        <Pipeline />
        <Capabilities />

        {/* Explainability analytics */}
        <section className="section pt-0">
          <div className="container-x grid items-center gap-10 lg:grid-cols-2">
            <SectionHeading
              eyebrow="Explainability"
              title="Explanations a pathologist can read"
              description="Rather than pixel heatmaps, CellGraphFM quantifies which cell types interact and how much — the language of tumour architecture and immune infiltration. Hover the matrix to read contact counts between any two cell types."
            />
            <InteractionMatrix />
          </div>
        </section>

        <Dimensions />
        <Platform />

        {/* CTA */}
        <section className="section">
          <div className="container-x">
            <div className="glass relative overflow-hidden px-6 py-14 text-center sm:px-12">
              <div className="grid-overlay pointer-events-none absolute inset-0" aria-hidden />
              <div className="relative mx-auto max-w-2xl">
                <h2 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
                  Bring graph intelligence to your pathology data
                </h2>
                <p className="mx-auto mt-4 max-w-xl text-ink-secondary">
                  Modular, explainable, and open-source. Install the Python core,
                  plug in your foundation model, and reason over tissue as a graph.
                </p>
                <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
                  <a
                    href={REPO}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-full bg-accent px-5 py-2.5 text-sm font-semibold text-white transition-transform hover:-translate-y-0.5"
                  >
                    Get started on GitHub
                  </a>
                  <code className="rounded-full border border-line bg-white/[0.04] px-4 py-2 font-mono text-xs text-ink-secondary">
                    pip install -e &quot;.[torch]&quot;
                  </code>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>
      <Footer />
    </>
  );
}
