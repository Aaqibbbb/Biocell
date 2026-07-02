import { Logo } from "./Logo";

const REPO = "https://github.com/Aaqibbbb/Biocell";

export function Footer() {
  return (
    <footer className="mt-auto border-t border-line">
      <div className="container-x flex flex-col gap-6 py-10 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2.5">
          <Logo className="h-6 w-6" />
          <span className="text-sm font-medium">CellGraphFM</span>
          <span className="text-sm text-ink-muted">· MIT licensed</span>
        </div>
        <div className="flex items-center gap-6 text-sm text-ink-secondary">
          <a href={REPO} target="_blank" rel="noreferrer" className="hover:text-ink">
            Repository
          </a>
          <a href={`${REPO}/blob/main/docs/ROADMAP.md`} target="_blank" rel="noreferrer" className="hover:text-ink">
            Roadmap
          </a>
          <a href={`${REPO}/issues`} target="_blank" rel="noreferrer" className="hover:text-ink">
            Issues
          </a>
        </div>
      </div>
      <div className="container-x pb-8 text-xs text-ink-muted">
        Open-source research software. Ships no patient data; not a medical device.
      </div>
    </footer>
  );
}
