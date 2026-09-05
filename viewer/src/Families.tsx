import React, { useMemo, useState } from "react";
import { FAMILY_LABELS, type Manifest } from "./types";
import { thumbUrl } from "./Catalogue";
import { AppearanceThumb, useAppearance, type AppearanceRender } from "./Appearance";
import { formatMass, Icon, PageIntro } from "./SiteUI";

const DESC: Record<string, string> = {
  swing_single: "Single hinged leaf: residential, commercial, fire/egress, institutional, industrial, detention, storefront glass, heritage.",
  swing_double: "Pairs: french doors, panic pairs with mullion or vertical rods, double-egress, storefront pairs, barn pairs.",
  dutch: "Independently hinged upper and lower halves with a joining bolt.",
  saloon: "Double-acting spring-hinged leaves that swing both ways.",
  pivot: "Oversized leaves on floor pivots (center or offset), floor-spring closers.",
  sliding_single: "Pocket, barn (flat track), patio glass with hook lock, shoji/fusuma, cell and industrial sliders.",
  sliding_bypass: "Two or three overlapping leaves on parallel tracks (closet, mirrored, shoji).",
  bifold: "Two or four panels folding on a pivot with a guided free edge (coupled joints).",
  accordion: "Concertina partitions with 6–10 coupled panels.",
  revolving: "Three or four wings on a rotor inside a drum with a speed governor.",
  turnstile_tripod: "Waist-high ratcheting tripod (one-way enforced by the environment).",
  turnstile_fullheight: "Full-height rotor with bar wings in a cage.",
  garage_sectional: "Overhead sectional door with torsion-spring counterbalance (vertical-lift approximation).",
  garage_tiltup: "One-piece tilt-up door on offset pivots.",
  rollup: "Coiling steel curtains and grilles with counterbalance.",
  pet_door: "Cat to XL-dog flaps swinging both ways, optional magnet.",
  hatch_floor: "Cellar trapdoors, deck and utility hatches with ring pulls, bolts, gas struts.",
  hatch_ceiling: "Attic/roof hatches pushed up through a curb.",
  ship_watertight: "Marine doors on a coaming with dogging levers or a central wheel.",
  vault: "Very heavy leaves with lever-driven bolt-work and crane hinges.",
  blast: "Heavy protective steel leaves with lever-operated bolts. Geometry does not imply a certified blast rating.",
  gate_swing: "Picket, chain-link, wrought iron, pool-safety and ranch gates with gravity latches, slide bolts, hasps.",
  gate_sliding: "Cantilever and track sliding gates.",
  baby_gate: "Pressure-mounted child gates with lift-pin latches.",
  stall: "Toilet partition doors on gravity hinges with slide latches.",
  strip_curtain: "Rows of overlapping PVC strips.",
  cold_storage: "Walk-in cooler/freezer doors: cam-lift hinges, gaskets, inside-release handles.",
  automatic_sliding: "Sensor-driven single/bi-parting sliders with breakout.",
  automatic_swing: "Low-energy and full-energy swing operators.",
  elevator: "Center- and side-opening landing doors with interlock and call button.",
};

const GROUPS: Record<string, string[]> = {
  "Swing & pivot": ["swing_single", "swing_double", "dutch", "saloon", "pivot", "automatic_swing"],
  "Slide & fold": ["sliding_single", "sliding_bypass", "bifold", "accordion", "automatic_sliding", "elevator"],
  "Rotate & lift": ["revolving", "turnstile_tripod", "turnstile_fullheight", "garage_sectional", "garage_tiltup", "rollup"],
  "Specialized access": ["hatch_floor", "hatch_ceiling", "ship_watertight", "vault", "blast", "gate_swing", "gate_sliding", "baby_gate", "stall", "strip_curtain", "cold_storage"],
};

export function Families({ manifest, supplementaryCount = 0 }: { manifest: Manifest; supplementaryCount?: number }) {
  const appearance = useAppearance();
  const [group, setGroup] = useState("All families");
  const photos = useMemo(() => {
    const result = new Map<string, AppearanceRender>();
    for (const r of appearance?.renders ?? []) if (r.image && (!result.has(r.door_id) || r.quality === "photo")) result.set(r.door_id, r);
    return result;
  }, [appearance]);
  const byFam = useMemo(() => {
    const result = new Map<string, typeof manifest.doors>();
    for (const d of manifest.doors) if (!d.error) result.set(d.family, [...(result.get(d.family) ?? []), d]);
    return result;
  }, [manifest]);
  const fams = manifest.families.filter((f) => byFam.has(f) && (group === "All families" || GROUPS[group].includes(f))).sort((a, b) => byFam.get(b)!.length - byFam.get(a)!.length);
  return <div className="page-shell families-page">
    <PageIntro eyebrow="The mechanics of access" title="One collection. Many ways in." aside={<a href="#/" className="button">View all doors <Icon name="arrow" /></a>}><p>Explore {manifest.families.length} motion families, from everyday hinges and sliding tracks to marine hatches, folding partitions, and revolving entrances.</p></PageIntro>
    <div className="family-navigation"><div className="category-tabs" aria-label="Motion categories">{["All families", ...Object.keys(GROUPS)].map((g) => <button key={g} aria-pressed={group === g} onClick={() => setGroup(g)}>{g}</button>)}</div><span>{fams.length} families</span></div>
    <div className="families">{fams.map((f) => {
      const ds = byFam.get(f)!;
      const rep = ds.find((d) => photos.get(d.id)?.quality === "photo") ?? ds.find((d) => photos.has(d.id)) ?? ds[0];
      const ops = new Set(ds.map((d) => d.operator)).size;
      const locks = new Set(ds.map((d) => d.lock)).size;
      const masses = ds.map((d) => d.mass_kg);
      return <a className="famcard" key={f} href={`#/?family=${f}`}><div className="family-image"><AppearanceThumb render={photos.get(rep.id)} fallback={thumbUrl(rep)} alt={`${FAMILY_LABELS[f] ?? f}: ${rep.use_case}`} /><span className="family-count">{ds.length} doors</span></div><div className="body"><div className="family-title"><h2>{FAMILY_LABELS[f] ?? f}</h2><Icon name="arrow" /></div><p>{DESC[f]}</p><div className="family-facts"><span><b>{ops}</b> {ops === 1 ? "operator" : "operators"}</span><span><b>{locks}</b> {locks === 1 ? "lock type" : "lock types"}</span><span><b>{formatMass(Math.min(...masses))}–{formatMass(Math.max(...masses))}</b> kg</span></div></div></a>;
    })}</div>
    {supplementaryCount > 0 && <aside className="catalogue-note"><div><strong>Supplementary pet-door collection</strong><p>{supplementaryCount} standalone pet flaps are available separately for download. They are outside the robot and human benchmark.</p></div><a href="#/pets">Browse pet doors <Icon name="arrow" size={16} /></a></aside>}
    <aside className="catalogue-note"><Icon name="door" /><p>Each family contains different materials, dimensions, hardware, and conditions. These are simulation models with documented approximations.</p><a href="#/about">Read the methodology <Icon name="arrow" size={16} /></a></aside>
  </div>;
}
