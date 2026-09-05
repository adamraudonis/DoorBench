import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

// Dev-only: serve ../assets at /assets (the generated dataset) and ../results at /results (benchmark results) in place.
function serveAssets(): Plugin {
  const roots: Record<string, string> = { "/assets/": path.resolve(import.meta.dirname, "..", "assets"), "/results/": path.resolve(import.meta.dirname, "..", "results") };
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
        if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) return next();
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

export default defineConfig({
  base: "./",
  plugins: [react(), serveAssets(), saveSnapshots()],
  build: { outDir: "dist", assetsDir: "static", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
});
