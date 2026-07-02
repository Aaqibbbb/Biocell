import type { Metadata, Viewport } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "CellGraphFM — Cellular Graph Intelligence for Pathology",
    template: "%s · CellGraphFM",
  },
  description:
    "A foundation-model platform that represents tissue as interacting biological graphs — not pixels — for diagnosis, prognosis, biomarker discovery, and scientific reasoning.",
  keywords: [
    "computational pathology",
    "graph neural networks",
    "foundation models",
    "digital pathology",
    "whole slide imaging",
    "cell graphs",
  ],
  authors: [{ name: "CellGraphFM" }],
  metadataBase: new URL("https://cellgraphfm.vercel.app"),
  openGraph: {
    title: "CellGraphFM — Cellular Graph Intelligence for Pathology",
    description:
      "Tissue as an evolving biological graph. A modular, explainable graph foundation model for computational pathology.",
    type: "website",
  },
};

export const viewport: Viewport = {
  themeColor: "#07080b",
  colorScheme: "dark",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${GeistSans.variable} ${GeistMono.variable} h-full`}
    >
      <body className="min-h-full flex flex-col antialiased">{children}</body>
    </html>
  );
}
