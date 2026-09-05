import React, { useEffect, useMemo, useState } from "react";
import type { Manifest } from "./types";
import { Catalogue } from "./Catalogue";
import { Families } from "./Families";
import { DoorView } from "./DoorView";
import { About } from "./About";
import { Results } from "./Results";
import { Review } from "./Review";
import { MotionLab } from "./MotionLab";
import { DATASET, Icon, REPOSITORY, SiteFooter } from "./SiteUI";

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
    if (h.startsWith("/families")) return { page: "families" };
    if (h.startsWith("/about")) return { page: "about" };
    if (h.startsWith("/results")) return { page: "results" };
    if (h.startsWith("/motions")) return { page: "motions" };
    if (h.startsWith("/review")) return { page: "review" };
    return { page: "catalogue", query: h.includes("?") ? h.split("?")[1] : "" };
  }, [hash]);
  useEffect(() => {
    document.getElementById("main-content")?.scrollTo({ top: 0, behavior: "instant" });
  }, [route.page, route.id]);
  return (
    <div className="app">
      <a className="skip-link" href="#main-content" onClick={(e) => { e.preventDefault(); document.getElementById("main-content")?.focus(); }}>Skip to content</a>
      <header className="topbar">
        <a className="brand" href="#/" aria-label="DoorBench home"><span className="brand-mark"><Icon name="door" size={24} /></span>DoorBench<span className="brand-tag">RESEARCH</span></a>
        <nav aria-label="Main navigation">
          {[["catalogue", "#/", "Catalogue"], ["families", "#/families", "Door types"], ["results", "#/results", "Results"], ["motions", "#/motions", "Motion Lab"], ["review", "#/review", "Review"], ["about", "#/about", "About"]].map(([page, href, label]) => <a key={page} href={href} className={route.page === page ? "active" : ""} aria-current={route.page === page ? "page" : undefined}>{label}</a>)}
        </nav>
        <div className="header-links"><a className="header-code" href={REPOSITORY} target="_blank" rel="noreferrer">GitHub <Icon name="external" size={13} /></a><a className="source-link" href={DATASET} target="_blank" rel="noreferrer">Get the dataset <Icon name="external" size={15} /></a></div>
      </header>
      <main id="main-content" tabIndex={-1} className={`content content-${route.page}`}>
        {err && <div className="empty-state"><Icon name="door" size={36} /><h1>The catalogue couldn’t load.</h1><p>Refresh the page to try again. If you’re running locally, generate the dataset first.</p><code>{err}</code><button onClick={() => window.location.reload()}>Try again</button></div>}
        {!manifest && !err && <div className="loading"><span className="loading-dot" />Loading the door collection…</div>}
        {manifest && route.page === "catalogue" && <Catalogue manifest={manifest} query={route.query ?? ""} />}
        {manifest && route.page === "families" && <Families manifest={manifest} />}
        {manifest && route.page === "door" && <DoorView manifest={manifest} id={route.id!} query={route.query ?? ""} />}
        {manifest && route.page === "about" && <About manifest={manifest} />}
        {manifest && route.page === "results" && <Results manifest={manifest} />}
        {manifest && route.page === "review" && <Review manifest={manifest} />}
        {manifest && route.page === "motions" && <MotionLab manifest={manifest} />}
        {manifest && !["door", "review"].includes(route.page) && <SiteFooter />}
      </main>
    </div>
  );
}
