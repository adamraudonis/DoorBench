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

export default defineConfig({
  base: "./",
  plugins: [react(), serveAssets()],
  build: { outDir: "dist", assetsDir: "static", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
});
