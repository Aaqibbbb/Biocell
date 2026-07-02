/** CellGraphFM mark — a small cell-graph motif (nodes + edges). */
export function Logo({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 32 32"
      className={className}
      fill="none"
      aria-hidden="true"
    >
      <g stroke="url(#lg)" strokeWidth="1.4" strokeLinecap="round" opacity="0.9">
        <path d="M9 22 L16 9 M16 9 L24 18 M9 22 L24 18 M16 9 L10 12 M24 18 L20 25" />
      </g>
      <g fill="url(#lg)">
        <circle cx="16" cy="9" r="3.1" />
        <circle cx="9" cy="22" r="2.5" />
        <circle cx="24" cy="18" r="2.5" />
        <circle cx="10" cy="12" r="1.7" />
        <circle cx="20" cy="25" r="1.7" />
      </g>
      <defs>
        <linearGradient id="lg" x1="6" y1="6" x2="26" y2="26" gradientUnits="userSpaceOnUse">
          <stop stopColor="#8fb6f2" />
          <stop offset="1" stopColor="#3987e5" />
        </linearGradient>
      </defs>
    </svg>
  );
}
