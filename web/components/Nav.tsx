import { Logo } from "./Logo";

const LINKS = [
  { href: "#platform", label: "Platform" },
  { href: "#pipeline", label: "Pipeline" },
  { href: "#capabilities", label: "Capabilities" },
  { href: "#research", label: "Research" },
];

const REPO = "https://github.com/Aaqibbbb/Biocell";

export function Nav() {
  return (
    <header className="glass-nav sticky top-0 z-50">
      <nav className="container-x flex h-16 items-center justify-between">
        <a href="#top" className="flex items-center gap-2.5">
          <Logo className="h-7 w-7" />
          <span className="text-[0.95rem] font-semibold tracking-tight">
            CellGraphFM
          </span>
        </a>

        <div className="hidden items-center gap-8 md:flex">
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="text-sm text-ink-secondary transition-colors hover:text-ink"
            >
              {l.label}
            </a>
          ))}
        </div>

        <a
          href={REPO}
          target="_blank"
          rel="noreferrer"
          className="rounded-full border border-line-strong bg-white/5 px-4 py-1.5 text-sm font-medium text-ink transition-colors hover:bg-white/10"
        >
          GitHub
        </a>
      </nav>
    </header>
  );
}
