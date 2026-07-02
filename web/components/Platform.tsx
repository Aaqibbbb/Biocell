import { SectionHeading } from "./SectionHeading";

const FEATURES: Array<{ label: string; status: "Live" | "Planned" }> = [
  { label: "Interactive 3D graph explorer", status: "Live" },
  { label: "Explainable interaction analytics", status: "Live" },
  { label: "Open API & self-hosting (MIT)", status: "Live" },
  { label: "WSI viewer — pan · zoom · annotate · ROI", status: "Planned" },
  { label: "Telepathology & real-time collaboration", status: "Planned" },
  { label: "LIS / PACS / HL7 integration", status: "Planned" },
  { label: "Report generation from the viewer", status: "Planned" },
  { label: "HIPAA · 21 CFR Part 11 · IEC 62304", status: "Planned" },
];

function Dot({ status }: { status: "Live" | "Planned" }) {
  const live = status === "Live";
  return (
    <span
      className="mt-1.5 h-2 w-2 shrink-0 rounded-full"
      style={{ background: live ? "#199e70" : "rgba(255,255,255,0.22)" }}
      aria-hidden
    />
  );
}

export function Platform() {
  return (
    <section id="platform" className="section">
      <div className="container-x grid items-start gap-10 lg:grid-cols-2">
        <SectionHeading
          eyebrow="Platform"
          title="An open cloud layer for digital pathology"
          description="CellGraphFM is the open-source AI and image-analysis layer that runs on top of any whole-slide scanner. It handles the software — detection, embedding, graph reasoning, and explanation — while integration, viewing, and collaboration build out on top."
        />

        <ul className="glass divide-y divide-[color:var(--color-line)] p-2">
          {FEATURES.map((f) => (
            <li key={f.label} className="flex items-start gap-3 px-4 py-3">
              <Dot status={f.status} />
              <span className="flex-1 text-sm text-ink-secondary">{f.label}</span>
              <span
                className="chip"
                style={
                  f.status === "Live"
                    ? { color: "#7fe0b8", borderColor: "rgba(25,158,112,0.4)" }
                    : undefined
                }
              >
                {f.status}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
