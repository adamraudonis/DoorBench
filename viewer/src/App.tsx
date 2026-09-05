import React, { Suspense, lazy, useEffect, useMemo, useState } from "react";
import type { Manifest } from "./types";
import { Catalogue } from "./Catalogue";
import { Families } from "./Families";
import { DoorView } from "./DoorView";
import { About } from "./About";

// Physics playground (MuJoCo WebAssembly): its own chunk, fetched only on #/playground so the door page stays light.
const Playground = lazy(() => import("./Playground").then((m) => ({ default: m.Playground })));

export const ASSETS = "./assets";

function useHash() {
  const [hash, setHash] = useState(window.location.hash || "#/");
  useEffect(() => {
    const on = () => setHash(window.location.hash || "#/");
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return hash;
}

export function App() {
  const hash = useHash();
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    fetch(`${ASSETS}/manifest.json`).then((r) => { if (!r.ok) throw new Error(`manifest ${r.status}`); return r.json(); }).then(setManifest).catch((e) => setErr(String(e)));
  }, []);
  const route = useMemo(() => {
    const h = hash.replace(/^#/, "");
    if (h.startsWith("/door/")) return { page: "door", id: h.slice(6).split("?")[0], query: h.includes("?") ? h.split("?")[1] : "" };
    if (h.startsWith("/playground")) return { page: "playground", id: h.slice("/playground".length).replace(/^\//, "").split("?")[0] || undefined, query: h.includes("?") ? h.split("?")[1] : "" };
    if (h.startsWith("/families")) return { page: "families" };
    if (h.startsWith("/about")) return { page: "about" };
    return { page: "catalogue", query: h.includes("?") ? h.split("?")[1] : "" };
  }, [hash]);
  return (
    <div className="app">
      <div className="topbar">
        <a className="brand" href="#/">DoorBench</a>
        <nav>
          <a href="#/" className={route.page === "catalogue" ? "active" : ""}>Catalogue</a>
          <a href="#/families" className={route.page === "families" ? "active" : ""}>Door types</a>
          <a href={route.page === "door" && route.id ? `#/playground/${route.id}` : "#/playground"} className={route.page === "playground" ? "active" : ""} title="Run the door in MuJoCo (WebAssembly) and tune its physical constants live">Physics playground</a>
          <a href="#/about" className={route.page === "about" ? "active" : ""}>About &amp; usage</a>
        </nav>
        <div className="spacer" />
        {manifest && <span style={{ color: "var(--muted)", fontSize: 12 }}>{manifest.n_doors} doors · {manifest.n_signed_off} signed off · v{manifest.version}</span>}
        <a href="https://github.com/adamraudonis/DoorBench" target="_blank" rel="noreferrer">GitHub</a>
      </div>
      <div className="content">
        {err && <div className="err">Could not load manifest: {err}. Run <code>python scripts/generate_dataset.py</code> first.</div>}
        {!manifest && !err && <div className="loading">Loading manifest…</div>}
        {manifest && route.page === "catalogue" && <Catalogue manifest={manifest} query={route.query ?? ""} />}
        {manifest && route.page === "families" && <Families manifest={manifest} />}
        {manifest && route.page === "door" && <DoorView manifest={manifest} id={route.id!} query={route.query ?? ""} />}
        {manifest && route.page === "playground" && <Suspense fallback={<div className="loading">Loading the playground…</div>}><Playground manifest={manifest} id={route.id} query={route.query ?? ""} /></Suspense>}
        {manifest && route.page === "about" && <About manifest={manifest} />}
      </div>
    </div>
  );
}
