// Lazy loader for the official MuJoCo WebAssembly bindings (`@mujoco/mujoco`, Google DeepMind, MuJoCo 3.12 = the
// version the dataset's QA ran on) and the staging of a door's MJCF + hardware meshes into the module's virtual FS.
//
// The 9.7 MB wasm is a separate Vite asset that is only requested when the playground route mounts (dynamic
// imports below); the single-threaded build is used because GitHub Pages cannot send the COOP/COEP headers the
// multi-threaded build needs.
import type { MainModule } from "@mujoco/mujoco";

/** Reads a dataset asset relative to the assets root ("hardware/x.obj", "doors/<id>/door.xml"). */
export type AssetReader = (rel: string) => Promise<Uint8Array>;

let modulePromise: Promise<MainModule> | null = null;

export function loadMujocoModule(): Promise<MainModule> {
  if (!modulePromise) {
    modulePromise = (async () => {
      const { default: factory } = await import("@mujoco/mujoco");
      const inBrowser = typeof window !== "undefined" && typeof document !== "undefined";
      // browser: Vite hands us the hashed asset URL; bun / node: emscripten resolves mujoco.wasm next to mujoco.js itself
      const wasmUrl = inBrowser ? (await import("@mujoco/mujoco/mujoco.wasm?url")).default : null;
      const mj = await factory(wasmUrl ? { locateFile: (p: string) => (p.endsWith(".wasm") ? wasmUrl : p) } : {});
      mkdirp(mj, "/assets/hardware");
      mkdirp(mj, "/assets/doors");
      return mj;
    })();
    modulePromise.catch(() => { modulePromise = null; });
  }
  return modulePromise;
}

/** Fetch-based reader for the site (dev server serves ../assets, the Pages build ships them next to the app). */
export function fetchReader(base: string): AssetReader {
  return async (rel) => {
    const r = await fetch(`${base}/${rel}`);
    if (!r.ok) throw new Error(`${rel}: HTTP ${r.status}`);
    return new Uint8Array(await r.arrayBuffer());
  };
}

export function mkdirp(mj: MainModule, path: string) {
  const parts = path.split("/").filter(Boolean);
  let cur = "";
  for (const p of parts) {
    cur += "/" + p;
    try { mj.FS.mkdir(cur); } catch { /* exists */ }
  }
}

function exists(mj: MainModule, path: string): boolean {
  try { mj.FS.stat(path); return true; } catch { return false; }
}

/** Mesh files referenced by an MJCF (the exporter writes `<mesh name=".." file="<key>.obj"/>` with meshdir=../../hardware). */
export function meshFiles(xml: string): string[] {
  return Array.from(xml.matchAll(/<mesh\b[^>]*\bfile="([^"]+)"/g), (m) => m[1]);
}

export const FS_DOOR_DIR = (id: string) => `/assets/doors/${id}`;

/** Write door.xml and every mesh it references into the virtual FS.  Returns the FS path of the XML.  Meshes are
 *  fetched once per session (they are shared between doors).  Missing meshes surface as a clear error rather than a
 *  MuJoCo compile failure. */
export async function stageDoor(mj: MainModule, id: string, xml: string, reader: AssetReader): Promise<string> {
  const dir = FS_DOOR_DIR(id);
  mkdirp(mj, dir);
  const missing: string[] = [];
  await Promise.all(meshFiles(xml).map(async (f) => {
    const p = `/assets/hardware/${f}`;
    if (exists(mj, p)) return;
    try { mj.FS.writeFile(p, await reader(`hardware/${f}`)); } catch (e) { missing.push(`${f} (${(e as Error).message})`); }
  }));
  if (missing.length) throw new Error(`mesh assets missing: ${missing.join(", ")}`);
  mj.FS.writeFile(`${dir}/door.xml`, xml);
  return `${dir}/door.xml`;
}

/** Human-readable text of a MuJoCo load / compile error thrown by the bindings. */
export function errorText(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (typeof e === "string") return e;
  if (typeof e === "number") return `MuJoCo aborted (wasm exception ${e})`;
  try { return JSON.stringify(e); } catch { return String(e); }
}
