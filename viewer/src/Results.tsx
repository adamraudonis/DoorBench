import React, { useMemo, useState } from "react";
import { FAMILY_LABELS, type Manifest } from "./types";
import { RESULTS, outcomeClass, useResultsIndex, type ResultEntry } from "./ResultBadges";

const pct = (x: number | null | undefined, d = 1) => (x === null || x === undefined ? "–" : `${(100 * x).toFixed(d)} %`);
const secs = (x: number | null | undefined) => (x === null || x === undefined ? "–" : `${x.toFixed(1)} s`);
const TASK_ORDER = ["open_and_traverse", "unlock_open_traverse", "hold_and_pass", "push_through", "traverse_open", "open_only", "close", "peek", "locked_recognize"];
const LOCK_LABEL: Record<string, string> = { unlocked: "unlocked / no lock", locked_releasable: "locked, robot-side release", locked_no_release: "locked, no release" };

function Bar({ n, d, cls = "ok" }: { n: number; d: number; cls?: string }) {
  const w = d > 0 ? (100 * n) / d : 0;
  return <div className="rbar" title={`${n} / ${d}`}><div className={`fill ${cls}`} style={{ width: `${w}%` }} /><span>{n} / {d}</span></div>;
}

export function Results({ manifest }: { manifest: Manifest }) {
  const idx = useResultsIndex();
  const [sel, setSel] = useState<string>("");
  const rows = useMemo(() => (idx ? [...idx.results].sort((a, b) => Number(b.leaderboard) - Number(a.leaderboard) || b.doors_solved - a.doors_solved || b.success_rate - a.success_rate) : []), [idx]);
  const total = idx?.n_doors_total ?? manifest.n_doors;
  const doors = manifest.doors.filter((d) => !d.error);
  const current: ResultEntry | undefined = rows.find((r) => r.file === sel) ?? rows[0];
  const families = useMemo(() => {
    const cnt = new Map<string, number>();
    for (const d of doors) cnt.set(d.family, (cnt.get(d.family) ?? 0) + 1);
    return [...cnt.entries()].sort((a, b) => b[1] - a[1]);
  }, [doors]);
  if (idx === undefined) return <div className="loading">Loading results…</div>;
  if (idx === null || !rows.length) {
    return (
      <div className="about">
        <h1>Benchmark results</h1>
        <p>No <code>results/index.json</code> found. Run a baseline and build the index:</p>
        <pre>{`doorbench benchmark run --policy scripted_hand --doors all --seeds 3 --workers 8 --out results/scripted_hand.json\npython scripts/build_results_index.py`}</pre>
      </div>
    );
  }
  const fams = families.map(([f]) => f);
  const tasks = TASK_ORDER.filter((t) => rows.some((r) => r.by_task[t]));
  const locks = ["unlocked", "locked_releasable", "locked_no_release"].filter((k) => rows.some((r) => r.by_lock_state[k]));
  return (
    <div className="results">
      <div className="about" style={{ paddingBottom: 0 }}>
        <h1 style={{ margin: "8px 0" }}>Benchmark results</h1>
        <p style={{ color: "var(--muted)", marginTop: 0 }}>
          Every row is one full run of a policy over the DoorBench doors, written by <code>doorbench benchmark run</code>, validated against <code>results/schema.json</code> and committed under <code>results/</code>.
          A door counts as <b>solved</b> when the policy succeeded on <b>every</b> seed of its own task (<code>spec.task</code>: open and traverse, unlock, hold against the closer, push through, close, peek, recognise a locked door …) without damaging it.
          Seed 0 is the nominal door; seeds ≥ 1 randomise friction, damping, closer stiffness and masses. Submit your own run: <a href="https://github.com/adamraudonis/DoorBench/blob/main/docs/SUBMITTING.md" target="_blank" rel="noreferrer">docs/SUBMITTING.md</a>.
        </p>
        <div className="stat-row">
          {rows.filter((r) => r.leaderboard).map((r) => (
            <div className="stat" key={r.file}><div className="n">{r.doors_solved} <span style={{ fontSize: 14, color: "var(--muted)" }}>/ {total}</span></div><div className="l">{r.policy.replace(/_/g, " ")}</div></div>
          ))}
        </div>
      </div>

      <div className="rsection">
        <h2>Leaderboard</h2>
        <div className="tablewrap">
          <table className="rtable">
            <thead><tr><th>policy</th><th>embodiment</th><th>simulator</th><th>tier</th><th>doors</th><th>seeds</th><th>solved (all seeds)</th><th>episode success</th><th>damage</th><th>median time-to-traverse</th><th>date</th><th>commit</th><th>run</th></tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.file} className={r.file === current?.file ? "sel" : ""} onClick={() => setSel(r.file)}>
                  <td><b>{r.policy}</b>{!r.leaderboard && <span className="chip" style={{ marginLeft: 6 }}>partial</span>}<div className="desc">{r.description}</div></td>
                  <td>{r.embodiment.replace(/_/g, " + ")}</td>
                  <td>{r.simulator} {r.simulator_version ?? ""}</td>
                  <td>{r.tier}</td>
                  <td>{r.n_doors}</td>
                  <td>{r.seeds.length}</td>
                  <td style={{ minWidth: 160 }}><Bar n={r.doors_solved} d={total} /></td>
                  <td>{pct(r.success_rate)}</td>
                  <td>{pct(r.damage_rate)}</td>
                  <td>{secs(r.median_time_to_pass_s)}</td>
                  <td>{r.date}</td>
                  <td><code>{(r.commit ?? "").slice(0, 8)}</code></td>
                  <td><a href={`${RESULTS}/${r.file}`} target="_blank" rel="noreferrer">json</a></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rsection">
        <h2>Per family <span className="sub">doors solved on every seed / doors in the family</span></h2>
        <div className="tablewrap">
          <table className="rtable fam">
            <thead><tr><th>family</th><th>n</th>{rows.map((r) => <th key={r.file}>{r.policy.replace(/_/g, " ")}</th>)}</tr></thead>
            <tbody>
              {fams.map((f) => (
                <tr key={f}>
                  <td><a href={`#/?family=${f}`}>{FAMILY_LABELS[f] ?? f}</a></td>
                  <td>{families.find((x) => x[0] === f)?.[1]}</td>
                  {rows.map((r) => { const g = r.by_family[f]; return <td key={r.file}>{g ? <Bar n={g.doors_solved} d={g.n_doors} /> : <span className="muted">–</span>}</td>; })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="rsection twocol">
        <div>
          <h2>Per task</h2>
          <div className="tablewrap">
            <table className="rtable">
              <thead><tr><th>task</th><th>n</th>{rows.map((r) => <th key={r.file}>{r.policy.replace(/_/g, " ")}</th>)}</tr></thead>
              <tbody>
                {tasks.map((t) => (
                  <tr key={t}><td>{t.replace(/_/g, " ")}</td><td>{Math.max(...rows.map((r) => r.by_task[t]?.n_doors ?? 0))}</td>
                    {rows.map((r) => { const g = r.by_task[t]; return <td key={r.file}>{g ? `${g.doors_solved} (${pct(g.success_rate, 0)})` : "–"}</td>; })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h2>Per lock state / difficulty</h2>
          <div className="tablewrap">
            <table className="rtable">
              <thead><tr><th>group</th><th>n</th>{rows.map((r) => <th key={r.file}>{r.policy.replace(/_/g, " ")}</th>)}</tr></thead>
              <tbody>
                {locks.map((k) => (
                  <tr key={k}><td>{LOCK_LABEL[k] ?? k}</td><td>{Math.max(...rows.map((r) => r.by_lock_state[k]?.n_doors ?? 0))}</td>
                    {rows.map((r) => { const g = r.by_lock_state[k]; return <td key={r.file}>{g ? `${g.doors_solved} (${pct(g.success_rate, 0)})` : "–"}</td>; })}
                  </tr>
                ))}
                {["1", "2", "3", "4", "5"].filter((k) => rows.some((r) => r.by_difficulty[k])).map((k) => (
                  <tr key={k}><td>difficulty L{k}</td><td>{Math.max(...rows.map((r) => r.by_difficulty[k]?.n_doors ?? 0))}</td>
                    {rows.map((r) => { const g = r.by_difficulty[k]; return <td key={r.file}>{g ? `${g.doors_solved} (${pct(g.success_rate, 0)})` : "–"}</td>; })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {current && (
        <div className="rsection">
          <h2>Per door <span className="sub">{current.policy.replace(/_/g, " ")} · {current.scenarios.join(", ")} scenario · {current.seeds.length} seed(s) · click a cell to open the door</span></h2>
          <div className="rtabs">
            {rows.map((r) => <button key={r.file} className={r.file === current.file ? "primary" : ""} onClick={() => setSel(r.file)}>{r.policy.replace(/_/g, " ")}</button>)}
            <span className="legend"><i className="ok" /> all seeds <i className="part" /> some seeds <i className="bad" /> none <i className="none" /> not evaluated</span>
          </div>
          <div className="doorgrid">
            {doors.map((d) => {
              const s = current.doors[d.id];
              return <a key={d.id} href={`#/door/${d.id}`} className={`cell ${outcomeClass(s)}`} title={`${d.id} · ${FAMILY_LABELS[d.family] ?? d.family} · ${d.task.replace(/_/g, " ")} · L${d.difficulty}${d.lock_engaged ? " · locked" : ""}\n${s ? `${s[0]} / ${s[1]} seeds successful` : "not evaluated"}`} />;
            })}
          </div>
          <div className="rmeta">
            <span>{current.n_episodes} episodes · {current.outcomes && Object.entries(current.outcomes).map(([k, v]) => `${v} ${k}`).join(" · ")}</span>
            <span>time budget {typeof current.time_budget_s === "number" ? `${current.time_budget_s} s` : "20 s (40 s delayed egress)"} · {current.randomize ? "domain randomisation on seeds ≥ 1" : "nominal physics"} · {current.mean_wall_s !== null && current.mean_wall_s !== undefined ? `${current.mean_wall_s.toFixed(2)} s wall / episode` : ""}{current.wall_time_s ? ` · ${(current.wall_time_s / 60).toFixed(1)} min total` : ""}</span>
            {current.host && <span>{current.host}{current.label ? ` · ${current.label}` : ""}</span>}
            {current.extra && Object.keys(current.extra).length > 0 && <span>{Object.entries(current.extra).map(([k, v]) => `${k}: ${String(v).slice(0, 40)}`).join(" · ")}</span>}
            <span>{current.policy_class}</span>
          </div>
        </div>
      )}
      <div className="about" style={{ paddingTop: 0 }}>
        <p style={{ color: "var(--muted)", fontSize: 12 }}>Index generated {idx.generated.slice(0, 10)} by <code>scripts/build_results_index.py</code>. Reproduce any row with <code>doorbench benchmark run --policy &lt;name&gt; --doors all --seeds 3</code>.</p>
      </div>
    </div>
  );
}
