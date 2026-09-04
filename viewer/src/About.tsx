import React from "react";
import type { Manifest } from "./types";

export function About({ manifest }: { manifest: Manifest }) {
  const doors = manifest.doors.filter((d) => !d.error);
  const fam = manifest.families.length;
  const ops = new Set(doors.map((d) => d.operator)).size;
  const locks = new Set(doors.map((d) => d.lock)).size;
  const slabs = new Set(doors.map((d) => d.leaf.slab)).size;
  const locked = doors.filter((d) => d.lock_engaged).length;
  return (
    <div className="about">
      <h1>DoorBench</h1>
      <p>A physics-grounded dataset and benchmark of <b>{doors.length} fully articulated doors</b> for training and evaluating humanoid robots in simulation. Every door ships as <b>MJCF</b> (MuJoCo, full fidelity with working latches, locks, closers and one-sided re-latching), <b>URDF</b> and <b>USD</b> (Isaac Sim / Isaac Lab), in three fidelity tiers (<code>full</code>, <code>simple</code>, <code>minimal</code>) for RL throughput.</p>
      <div className="stat-row">
        <div className="stat"><div className="n">{doors.length}</div><div className="l">doors</div></div>
        <div className="stat"><div className="n">{fam}</div><div className="l">kinematic families</div></div>
        <div className="stat"><div className="n">{ops}</div><div className="l">operator types</div></div>
        <div className="stat"><div className="n">{locks}</div><div className="l">lock types</div></div>
        <div className="stat"><div className="n">{slabs}</div><div className="l">slab constructions</div></div>
        <div className="stat"><div className="n">{locked}</div><div className="l">start locked</div></div>
        <div className="stat"><div className="n">{manifest.n_signed_off}</div><div className="l">QA signed off</div></div>
      </div>
      <h2>What is modelled</h2>
      <ul>
        <li><b>Mass & inertia</b> from slab build-ups (skins + core + stiles) calibrated to manufacturer door-weight tables, glazing at 2,500 kg/m³, hardware masses per item.</li>
        <li><b>Hinge friction</b> from a pin/thrust bearing load model (μ per bearing type & condition), rolling friction for tracks, air damping.</li>
        <li><b>Closers</b> sized per EN 1154 (spring preload + rate, asymmetric hydraulic damping, backcheck, hold-open), spring hinges, floor springs, gas struts, counterbalance springs.</li>
        <li><b>Latches</b> as real bodies: spring bolts with beveled strike lips (doors re-latch when slammed), deadbolts driven by thumbturns, panic touch bars, hooks, slide bolts, dogs, vault boltwork, keypad buttons, REX buttons, maglocks with breakaway.</li>
        <li><b>Locks</b>: privacy, keyed, deadbolts, chains, guards, padlocks, keypad codes, card readers, maglocks, delayed egress, interlocks — with "jiggle" backlash when locked and a per-door flag for whether the robot can release it from its side.</li>
        <li><b>Code compliance</b> flags (ADA §404 5 lbf, IBC §1010 30/15 lbf, panic 15 lbf) computed from the simulated forces.</li>
        <li><b>Damage thresholds</b> per material and hardware item, used by the benchmark labeller (dents, glass, operator yield, latch shear, slams, forced maglocks).</li>
      </ul>
      <h2>Using the dataset</h2>
      <pre>{`pip install -e .            # from the repo root
python -c "
from doorbench.benchmark import DoorEnv
env = DoorEnv('assets/doors/db0002_swing_single', tier='full')
env.reset()
for _ in range(600):
    env.apply_joint_torque(env.meta['operator_joint'], 3.0)   # turn the knob
    env.apply_joint_torque(env.meta['primary_joint'], 30.0)   # push the door
    env.step()
print(env.labels().to_dict())
"`}</pre>
      <p>MuJoCo: <code>python -m mujoco.viewer --mjcf assets/doors/&lt;id&gt;/scene.xml</code>. Isaac Lab: import <code>door.usda</code> (articulation root on the door prim; joint drives carry closer springs). URDF loaders: <code>door.urdf</code> with <code>&lt;mimic&gt;</code> couplings and <code>doorbench:</code> extension tags for springs.</p>
      <h2>Benchmark scenarios &amp; rewards</h2>
      <p>Every door carries one or more evaluation scenarios in <code>spec.json.benchmark</code>, split into two suites. The <b>core</b> suite needs nothing but the door and the robot and is the default for every benchmark run and every published table: <b>open &amp; traverse</b> (or <b>unlock &amp; traverse</b> / <b>locked-recognize</b> for locked doors), <b>open then close</b> and <b>close only</b>. The <b>human</b> suite is an advanced, opt-in tier with a simulated person, reported separately and never mixed into the core number: <b>hold open for a human</b>, <b>wait for a human</b> coming the other way, and <b>knock &amp; wait</b>. Each scenario fixes a randomisable start zone (seeded), the approach point, the handle targets (grip / push sites), the pass plane, the goal zone, an optional simulated-human path, a reward table (touch handle +1, unlatch +2, open +3, traverse +10, close behind +3, hold for human +5, collision with human −20, damage −10, slam −2, −0.05 / s), a time budget and an expected transit time derived from the door's mass, friction, closer and geometry. Open a door and press <b>Show evaluation</b> to see them; the formulas are in <code>docs/BENCHMARK.md</code>.</p>
      <h2>Benchmark labels</h2>
      <p>Every episode yields: touched door / operator, operator actuated, latch released, lock released, door opened / clear, robot passed through, door closed after, slammed, damaged (with event list), robot fell, hardware misuse, peak forces, time-to-touch/open/pass, energy, reward events, episode return and the scenario's success flag. Legacy tasks (open only, traverse open door, push-through, hold-and-pass, peek) are still accepted by <code>reset(task=...)</code>.</p>
      <h2>Provenance</h2>
      <p>Physics catalogs cite EN 1154:1996, ANSI/BHMA A156.2/A156.5, UL 305, ADA 2010 §404, IBC §1010, the Steel Door Institute, Knape &amp; Vogt door-weight tables, the USDA Wood Handbook and manufacturer catalogs (LCN, Norton, Von Duprin, Schlage, Kason, D&amp;D). All geometry is procedurally generated; Poly Haven CC0 textures are referenced for photoreal renders.</p>
      <p>Generated {manifest.generated} · seed {String((manifest as any).seed)} · MIT license.</p>
    </div>
  );
}
