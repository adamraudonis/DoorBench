import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import path from "node:path";

// Dev-only: serve ../assets at /assets so the viewer reads the generated dataset in place.
function serveAssets(): Plugin {
  const root = path.resolve(import.meta.dirname, "..", "assets");
  const types: Record<string, string> = { ".json": "application/json", ".obj": "text/plain", ".jpg": "image/jpeg", ".png": "image/png", ".xml": "text/xml", ".urdf": "text/xml", ".usda": "text/plain", ".glb": "model/gltf-binary" };
  return {
    name: "serve-assets",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !req.url.startsWith("/assets/")) return next();
        const rel = decodeURIComponent(req.url.split("?")[0].slice("/assets/".length));
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
