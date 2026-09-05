import React from "react";

export const REPOSITORY = "https://github.com/adamraudonis/DoorBench";
export const DOCS = `${REPOSITORY}/blob/master/docs`;
export const DATASET = "https://huggingface.co/datasets/adamraudonis/DoorBench";
export const formatMass = (kg: number) => kg.toFixed(kg < 1 ? 2 : kg < 10 ? 1 : 0);

export function Icon({ name = "arrow", size = 18 }: { name?: "arrow" | "external" | "search" | "filter" | "grid" | "close" | "door" | "check" | "code"; size?: number }) {
  const paths = {
    arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
    external: <path d="M14 4h6v6M20 4l-9 9M10 4H5a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5" />,
    search: <><circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 4.5 4.5" /></>,
    filter: <><path d="M4 7h16M4 17h16" /><circle cx="9" cy="7" r="2" fill="currentColor" /><circle cx="15" cy="17" r="2" fill="currentColor" /></>,
    grid: <><rect x="4" y="4" width="6" height="6" rx="1" /><rect x="14" y="4" width="6" height="6" rx="1" /><rect x="4" y="14" width="6" height="6" rx="1" /><rect x="14" y="14" width="6" height="6" rx="1" /></>,
    close: <path d="m6 6 12 12M6 18 18 6" />,
    door: <><path d="M4 21V4h15v17M7 21V2l10 3v16zM2 21h20" /><path d="M13 12v2" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    code: <path d="m8 6-6 6 6 6M16 6l6 6-6 6M14 3l-4 18" />,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

export function PageIntro({ eyebrow, title, children, aside }: { eyebrow: string; title: string; children: React.ReactNode; aside?: React.ReactNode }) {
  return <header className="page-intro"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><div className="page-lead">{children}</div></div>{aside && <div className="intro-aside">{aside}</div>}</header>;
}

export function SiteFooter() {
  return <footer className="site-footer"><a href="#/" className="footer-brand"><Icon name="door" />DoorBench</a><span>Open environments for robotics research.</span><div><a href="#/about">Documentation</a><a href={REPOSITORY} target="_blank" rel="noreferrer">GitHub <Icon name="external" size={13} /></a><a href="https://polyhaven.com/license" target="_blank" rel="noreferrer">Texture credits</a></div></footer>;
}
