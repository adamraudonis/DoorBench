import fs from 'node:fs';
import path from 'node:path';
import type {Plugin} from 'vite';

/** Only these named files are exposed, only by Vite's development server. */
export function humanReferenceDev(repoRoot: string, configuredRoot?: string): Plugin {
  const root = path.resolve(repoRoot, configuredRoot || 'out/human-reference/ceti-d02-o03');
  const files: Record<string, {root: string; name: string; type: string}> = {
    'contact-fit.json': {root, name: 'contact-fit.json', type: 'application/json'},
    'motion.json': {root, name: 'motion.json', type: 'application/json'},
    'normal-speed.mp4': {root, name: 'normal-speed.mp4', type: 'video/mp4'},
    'animation.glb': {root, name: 'animation.glb', type: 'model/gltf-binary'},
    'methodology.md': {root: path.join(repoRoot, 'docs'), name: 'HUMAN_REFERENCE.md', type: 'text/plain; charset=utf-8'},
  };
  return {name: 'local-human-reference', apply: 'serve', configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const prefix = '/__human-reference/';
      if (!req.url?.startsWith(prefix)) return next();
      if (req.method !== 'GET' && req.method !== 'HEAD') {res.statusCode = 405; res.end(); return;}
      const entry = files[req.url.split('?')[0].slice(prefix.length)];
      if (!entry) {res.statusCode = 404; res.end('Unknown local preview file.'); return;}
      try {
        const file = fs.realpathSync(path.join(entry.root, entry.name));
        if (!file.startsWith(fs.realpathSync(entry.root) + path.sep) || !fs.statSync(file).isFile()) throw new Error('Outside preview root.');
        res.setHeader('Content-Type', entry.type);
        res.setHeader('Content-Length', fs.statSync(file).size);
        res.setHeader('Cache-Control', 'no-store');
        res.setHeader('X-Content-Type-Options', 'nosniff');
        if (req.method === 'HEAD') {res.end(); return;}
        const stream = fs.createReadStream(file);
        stream.on('error', () => {res.destroy();});
        res.on('close', () => stream.destroy());
        stream.pipe(res);
      } catch {res.statusCode = 404; res.end('Local human preview is not exported yet.');}
    });
  }};
}
