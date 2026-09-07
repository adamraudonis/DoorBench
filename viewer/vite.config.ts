import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";
import {execFileSync} from "node:child_process";
import {localReviewDev} from "./localReviewDev.ts";
import { humanReferenceDev } from "./humanReferenceDev.ts";

// Dev-only: serve ../assets at /assets (the generated dataset) and ../results at /results (benchmark results) in place.
function serveAssets(): Plugin {
  const repoRoot = path.resolve(import.meta.dirname, "..");
  // Preview a verified publication directory without replacing a local dataset.
  const siteRoot = process.env.DOORBENCH_SITE_ROOT ? path.resolve(repoRoot, process.env.DOORBENCH_SITE_ROOT) : null;
  // Source checkouts keep generated media under out/; published sites keep it at the root.
  const mediaRoot = (name: string) => {
    const root = siteRoot ?? repoRoot;
    const published = path.join(root, name);
    return siteRoot && (fs.existsSync(published) || !fs.existsSync(path.join(root, ".git")))
      ? published : path.join(root, "out", name);
  };
  // Prepared release previews stay local; relative overrides use the repository root.
  const plannedWebRoot = path.resolve(repoRoot, process.env.DOORBENCH_PLANNED_WEB_ROOT || "out/planned-reference-web");
  const roots: Record<string, string> = { "/planned-references/": plannedWebRoot, "/reference-motions/": mediaRoot("reference-motions"), "/assets/": path.resolve(siteRoot ?? repoRoot, "assets"), "/results/": path.resolve(repoRoot, "results"), "/appearance/": mediaRoot("appearance") };
  const types: Record<string, string> = { ".json": "application/json", ".obj": "text/plain", ".jpg": "image/jpeg", ".png": "image/png", ".xml": "text/xml", ".urdf": "text/xml", ".usda": "text/plain", ".glb": "model/gltf-binary", ".md": "text/markdown" };
  return {
    name: "serve-assets",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const prefix = req.url ? Object.keys(roots).find((p) => req.url!.startsWith(p)) : undefined;
        if (!req.url || !prefix) return next();
        const root = roots[prefix];
        const rel = decodeURIComponent(req.url.split("?")[0].slice(prefix.length));
        const file = path.join(root, rel);
        if (!file.startsWith(root + path.sep) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) return next();
        res.setHeader("Content-Type", types[path.extname(file)] ?? "application/octet-stream");
        res.setHeader("Cache-Control", "no-cache");
        fs.createReadStream(file).pipe(res);
      });
    },
  };
}

// Dev-only: POST a PNG data URL to /__snapshot?name=<file>.png to save it under ../docs/media (how the viewer
// screenshots referenced by the docs are captured from the live page; never part of the built site).
function saveSnapshots(): Plugin {
  const dir = path.resolve(import.meta.dirname, "..", "docs", "media");
  return {
    name: "save-snapshots",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !req.url.startsWith("/__snapshot") || req.method !== "POST") return next();
        const name = path.basename(new URL(req.url, "http://localhost").searchParams.get("name") ?? "");
        if (!/^[\w.-]+\.png$/.test(name)) { res.statusCode = 400; res.end("name must be <file>.png"); return; }
        const chunks: Buffer[] = [];
        req.on("data", (c: Buffer) => chunks.push(c));
        req.on("end", () => {
          const m = /^data:image\/png;base64,([A-Za-z0-9+/=]+)$/.exec(Buffer.concat(chunks).toString("utf8"));
          if (!m) { res.statusCode = 400; res.end("expected a PNG data URL"); return; }
          fs.mkdirSync(dir, { recursive: true });
          const file = path.join(dir, name);
          fs.writeFileSync(file, Buffer.from(m[1], "base64"));
          res.setHeader("Content-Type", "text/plain");
          res.end(file);
        });
      });
    },
  };
}

// Use the shared repository root so reviews survive worktree changes and rebuilds.
const repoRoot=path.resolve(import.meta.dirname,"..");
const commonGit=execFileSync("git",["rev-parse","--path-format=absolute","--git-common-dir"],{cwd:repoRoot,encoding:"utf8"}).trim();
const reviewRoot=process.env.DOORBENCH_REVIEW_ROOT||path.join(path.dirname(commonGit),"out","local-reviews");
const reviewAssets=path.resolve(repoRoot,process.env.DOORBENCH_SITE_ROOT||".","assets");

export default defineConfig({
  base: "./",
  plugins: [localReviewDev(reviewRoot,reviewAssets), react(), serveAssets(), saveSnapshots(), humanReferenceDev(path.resolve(import.meta.dirname, ".."), process.env.DOORBENCH_HUMAN_REFERENCE_ROOT)],
  build: { outDir: "dist", assetsDir: "static", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
});
