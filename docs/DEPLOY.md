# Deploying the CellGraphFM web app to Vercel

The web frontend lives in [`web/`](../web) (Next.js 16 · React 19 · Tailwind 4 ·
React Three Fiber). It is a static, client-rendered site — no server runtime or
secrets required — so it deploys anywhere Next.js runs.

## Option A — Vercel dashboard (recommended, no token)

1. Go to <https://vercel.com/new> and **Import** the `Aaqibbbb/Biocell` repo.
2. **Set the Root Directory to `web`.** (This is the one setting that matters for
   a monorepo — Vercel then finds `web/package.json` and auto-detects Next.js.)
3. Leave the framework preset as **Next.js**; build command `next build` and the
   install command are auto-detected.
4. **Deploy.** Every push to `main` that touches `web/**` redeploys automatically.

## Option B — Vercel CLI

```bash
npm i -g vercel
cd web
vercel            # first run links/creates the project — set root dir = ./ (you are in web/)
vercel --prod     # promote to production
```

With a token (CI / non-interactive):

```bash
cd web
vercel pull --yes --environment=production --token "$VERCEL_TOKEN"
vercel build --prod --token "$VERCEL_TOKEN"
vercel deploy --prebuilt --prod --token "$VERCEL_TOKEN"
```

## Local development

```bash
cd web
npm install
npm run dev     # http://localhost:3000
npm run build   # production build (what CI runs)
npm run lint
```

CI builds the app on every `web/**` change via
[`.github/workflows/web.yml`](../.github/workflows/web.yml).
