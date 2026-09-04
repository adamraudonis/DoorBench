import React from "react";
import { FAMILY_LABELS, type Manifest } from "./types";
import { thumbUrl } from "./Catalogue";

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
  blast: "Blast-rated steel leaves with lever bolts.",
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

export function Families({ manifest }: { manifest: Manifest }) {
  const byFam = new Map<string, typeof manifest.doors>();
  for (const d of manifest.doors) { if (d.error) continue; if (!byFam.has(d.family)) byFam.set(d.family, []); byFam.get(d.family)!.push(d); }
  const fams = manifest.families.filter((f) => byFam.has(f)).sort((a, b) => byFam.get(b)!.length - byFam.get(a)!.length);
  return (
    <div>
      <div className="about" style={{ paddingBottom: 0 }}>
        <h1 style={{ margin: "8px 0" }}>Door types</h1>
        <p style={{ color: "var(--muted)", marginTop: 0 }}>{fams.length} kinematic families covering every human- or animal-passable door we could enumerate. Click a family to filter the catalogue; every door opens in the 3D viewer.</p>
      </div>
      <div className="families">
        {fams.map((f) => {
          const ds = byFam.get(f)!;
          const rep = ds.find((d) => d.signed_off && d.thumbs.length) ?? ds[0];
          const ops = new Set(ds.map((d) => d.operator)).size;
          const locks = new Set(ds.map((d) => d.lock)).size;
          const masses = ds.map((d) => d.mass_kg);
          return (
            <a className="famcard" key={f} href={`#/?family=${f}`}>
              <img src={thumbUrl(rep)} loading="lazy" alt={f} />
              <div className="body">
                <h3>{FAMILY_LABELS[f] ?? f} <span style={{ color: "var(--muted)", fontWeight: 400 }}>· {ds.length}</span></h3>
                <p>{DESC[f]}</p>
                <p>{ops} operator types · {locks} lock types · {Math.min(...masses).toFixed(0)}–{Math.max(...masses).toFixed(0)} kg · {ds.filter((d) => d.signed_off).length}/{ds.length} signed off</p>
              </div>
            </a>
          );
        })}
      </div>
    </div>
  );
}
