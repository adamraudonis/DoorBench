import { isPetDoor, resultsRespectEligibility } from "./collections";
import React, { useMemo, useState } from "react";
import { FAMILY_LABELS, type Manifest } from "./types";
import { DOCS, Icon, PageIntro } from "./SiteUI";
import { RESULTS, outcomeClass, useResultsIndex, type ResultEntry, type Suite, type SuiteBlock } from "./ResultBadges";

const pct = (x: number | null | undefined, d = 1) => (x === null || x === undefined ? "–" : `${(100 * x).toFixed(d)} %`);
const secs = (x: number | null | undefined) => (x === null || x === undefined ? "–" : `${x.toFixed(1)} s`);
const LOCK_LABEL: Record<string, string> = { unlocked: "unlocked / no lock", locked_releasable: "locked, robot-side release", locked_no_release: "locked, no release" };
const SCENARIO_LABEL: Record<string, string> = {
  open_only: "open for access", open_and_traverse: "open & traverse", open_then_close: "open, traverse, close behind", close_only: "close only (starts open)", unlock_and_traverse: "unlock & traverse", locked_recognize: "locked: recognise & stop",
  hold_open_for_human: "hold open for a person", wait_for_human: "wait for a person coming through", knock_and_wait: "knock & wait",
};

function Bar({ n, d, cls = "ok" }: { n: number; d: number; cls?: string }) {
  const w = d > 0 ? (100 * n) / d : 0;
  return <div className="rbar" title={`${n} / ${d}`}><div className={`fill ${cls}`} style={{ width: `${w}%` }} /><span>{n} / {d}</span></div>;
}

function sortRows(rows: ResultEntry[], suite: Suite): ResultEntry[] {
  return rows.filter((r) => r.suites[suite]).sort((a, b) => {
    const A = a.suites[suite]!, B = b.suites[suite]!;
    return Number(B.complete) - Number(A.complete) || B.doors_solved - A.doors_solved || B.success_rate - A.success_rate;
  });
}

export function Results({ manifest }: { manifest: Manifest }) {
  const idx = useResultsIndex();
  const [suite, setSuite] = useState<Suite>("core");
  const [sel, setSel] = useState<string>("");
  const rows = useMemo(() => (idx ? sortRows(idx.results, suite) : []), [idx, suite]);
  const doors = useMemo(() => manifest.doors.filter((d) => !d.error && !isPetDoor(d) && (suite === "core" || (d.benchmark?.human?.length ?? 0) > 0)), [manifest, suite]);
  const total = suite === "core" ? idx?.n_doors_total ?? manifest.n_doors : idx?.n_doors_human ?? doors.length;
  const current: ResultEntry | undefined = rows.find((r) => r.file === sel) ?? rows[0];
  const cur: SuiteBlock | undefined = current?.suites[suite];
  const families = useMemo(() => {
    const cnt = new Map<string, number>();
    for (const d of doors) cnt.set(d.family, (cnt.get(d.family) ?? 0) + 1);
    return [...cnt.entries()].sort((a, b) => b[1] - a[1]);
  }, [doors]);
  if (idx === undefined) return <div className="loading">Loading results…</div>;
  if (idx === null || !idx.results.length) {
    return (
      <div className="about">
        <h1>Benchmark results</h1>
        <p>No <code>results/index.json</code> found. Run a baseline and build the index:</p>
        <pre>{`doorbench benchmark run --policy scripted_hand --doors all --seeds 3 --workers 8 --out results/scripted_hand.json\npython scripts/build_results_index.py`}</pre>
      </div>
    );
  }
  if (!resultsRespectEligibility(idx, manifest)) return <div className="page-shell results"><PageIntro eyebrow="Recorded baselines" title="Results require the standard-door subset."><p>This archived index still includes supplementary assets or lacks verified eligibility metadata. Benchmark totals are hidden until the index is rebuilt from eligible source episodes.</p></PageIntro><p>Standalone pet doors are excluded from robot and human evaluation. <a href="#/pets">Their assets remain downloadable separately.</a></p></div>;
  const scenarios = (idx.suites?.[suite] ?? []).filter((s) => rows.some((r) => r.suites[suite]?.by_scenario[s]));
  const fams = families.map(([f]) => f);
  const locks = ["unlocked", "locked_releasable", "locked_no_release"].filter((k) => rows.some((r) => r.suites[suite]?.by_lock_state[k]));
  const g = (r: ResultEntry) => r.suites[suite]!;
  const suiteTabs = <div className="suite-selector"><div className="category-tabs" aria-label="Benchmark suite"><button aria-pressed={suite === "core"} onClick={() => { setSuite("core"); setSel(""); }}>Core suite <span>Default</span></button><button aria-pressed={suite === "human"} onClick={() => { setSuite("human"); setSel(""); }}>Human suite <span>Opt-in</span></button></div><p>{suite === "core" ? "Door-and-robot tasks across the standard-door collection. Supplementary pet doors are excluded. No simulated person." : `Shared-passage tasks with a simulated person, across ${idx.n_doors_human} eligible doors. Reported separately from core results.`}</p></div>;

  return (
    <div className="results page-shell">
      <PageIntro eyebrow="Evaluation / recorded baselines" title="Measure the whole interaction." aside={<a className="button" href={`${DOCS}/SUBMITTING.md`} target="_blank" rel="noreferrer">Submit a run <Icon name="external" size={15} /></a>}><p>Compare policies across mechanisms, scenarios, and physical variation. A solved door means every assigned scenario, on every evaluation seed.</p></PageIntro>
      <aside className="historical-notice"><span className="notice-mark">i</span><div><strong>Historical results · earlier dataset revision</strong><p>These runs predate the current geometry repairs and Blender appearance update. Scores describe the original run commit, not the latest door assets. Each row links to its recorded metadata.</p></div></aside>
      {rows.some(r => r.historical_subset?.applied) && <aside className="historical-notice"><span className="notice-mark">i</span><div><strong>Historical run · eligible-door subset</strong><p>Metrics were recomputed from original episodes after excluding standalone pet doors. This is not a new evaluation. Downloads retain the original run files; full-run wall time is unchanged.</p></div></aside>}
      {suiteTabs}
      <div className="result-summary">{rows.filter((r) => g(r).complete).map((r, i) => <button className={`result-stat ${r.file === current?.file ? "selected" : ""}`} key={r.file} onClick={() => setSel(r.file)}><span className="result-policy"><span className="rank">0{i + 1}</span>{r.policy.replace(/_/g, " ")}</span><strong>{g(r).doors_solved}<small> / {total}</small></strong><div className="summary-progress"><span style={{ width: `${total ? 100 * g(r).doors_solved / total : 0}%` }} /></div><span className="result-stat-footer">Doors solved <b>{pct(total ? g(r).doors_solved / total : 0)}</b></span></button>)}</div>
      <details className="evaluation-method"><summary>How to read these results</summary><p>A door counts as solved when the policy succeeds on every scenario assigned to that door in the chosen suite, on every seed, without damage. Seed 0 uses nominal parameters; later seeds randomize friction, damping, closer stiffness, mass, and start pose. A row marked “subset” did not evaluate the complete suite.</p><p>Episode success measures individual scenario-and-seed trials. It can be higher than the fraction of fully solved doors. Core and human suites use different task sets and are not combined.</p></details>

      <div className="rsection">
        <h2>Recorded runs <span className="sub">{suite === "core" ? "core suite" : "human suite (advanced, opt-in)"}</span></h2>
        {rows.length === 0 && <p className="muted">No {suite}-suite run yet.</p>}
        {rows.length > 0 && (
          <div className="tablewrap">
            <table className="rtable">
              <thead><tr><th>policy</th><th>embodiment</th><th>simulator</th><th>tier</th><th>doors</th><th>seeds</th><th>solved (every scenario, every seed)</th><th>episode success</th>{suite === "human" && <th>human collisions</th>}<th>damage</th><th>median time-to-traverse</th><th>date</th><th>commit</th><th>run</th></tr></thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.file} className={r.file === current?.file ? "sel" : ""} onClick={() => setSel(r.file)}>
                    <td><button className="table-policy" aria-pressed={r.file === current?.file} onClick={() => setSel(r.file)}>{r.policy.replace(/_/g, " ")}</button>{!g(r).complete && <span className="chip" style={{ marginLeft: 6 }} title={`door selection: ${r.door_selection ?? ""}`}>subset</span>}<div className="desc">{r.description}</div>{r.historical_subset?.applied && <div className="desc" title={r.historical_subset.note}>{r.historical_subset.excluded_n_doors} supplementary doors / {r.historical_subset.excluded_n_episodes} episodes excluded · historical subset</div>}</td>
                    <td>{r.embodiment.replace(/_/g, " + ")}</td>
                    <td>{r.simulator} {r.simulator_version ?? ""}</td>
                    <td>{r.tier}</td>
                    <td>{g(r).n_doors}</td>
                    <td>{r.seeds.length}</td>
                    <td style={{ minWidth: 160 }}><Bar n={g(r).doors_solved} d={g(r).complete ? total : g(r).n_doors} /></td>
                    <td>{pct(g(r).success_rate)}</td>
                    {suite === "human" && <td>{pct(g(r).human_collision_rate)}</td>}
                    <td>{pct(g(r).damage_rate)}</td>
                    <td>{secs(g(r).median_time_to_pass_s)}</td>
                    <td>{r.date}</td>
                    <td><code>{(r.commit ?? "").slice(0, 8)}</code></td>
                    <td><a href={`${RESULTS}/${r.file}`} target="_blank" rel="noreferrer">json</a></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {rows.length > 0 && (
        <div className="rsection">
          <h2>Per scenario <span className="sub">doors solved on every seed / doors listing the scenario (episode success)</span></h2>
          <div className="tablewrap">
            <table className="rtable">
              <thead><tr><th>scenario</th><th>n</th>{rows.map((r) => <th key={r.file}>{r.policy.replace(/_/g, " ")}</th>)}</tr></thead>
              <tbody>
                {scenarios.map((s) => (
                  <tr key={s}><td>{SCENARIO_LABEL[s] ?? s}<div className="desc">{s}</div></td><td>{Math.max(...rows.map((r) => g(r).by_scenario[s]?.n_doors ?? 0))}</td>
                    {rows.map((r) => { const x = g(r).by_scenario[s]; return <td key={r.file}>{x ? <Bar n={x.doors_solved} d={x.n_doors} /> : <span className="muted">–</span>}<div className="desc">{x ? `${pct(x.success_rate, 0)} of episodes${x.human_collision_rate !== null && x.human_collision_rate !== undefined ? ` · ${pct(x.human_collision_rate, 0)} collisions` : ""}` : ""}</div></td>; })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {rows.length > 0 && (
        <div className="rsection">
          <h2>Per family <span className="sub">doors solved / doors in the family{suite === "human" ? " with a human scenario" : ""}</span></h2>
          <div className="tablewrap">
            <table className="rtable fam">
              <thead><tr><th>family</th><th>n</th>{rows.map((r) => <th key={r.file}>{r.policy.replace(/_/g, " ")}</th>)}</tr></thead>
              <tbody>
                {fams.map((f) => (
                  <tr key={f}>
                    <td><a href={`#/?family=${f}`}>{FAMILY_LABELS[f] ?? f}</a></td>
                    <td>{families.find((x) => x[0] === f)?.[1]}</td>
                    {rows.map((r) => { const x = g(r).by_family[f]; return <td key={r.file}>{x ? <Bar n={x.doors_solved} d={x.n_doors} /> : <span className="muted">–</span>}</td>; })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {rows.length > 0 && (
        <div className="rsection">
          <h2>Per lock state / difficulty</h2>
          <div className="tablewrap">
            <table className="rtable">
              <thead><tr><th>group</th><th>n</th>{rows.map((r) => <th key={r.file}>{r.policy.replace(/_/g, " ")}</th>)}</tr></thead>
              <tbody>
                {locks.map((k) => (
                  <tr key={k}><td>{LOCK_LABEL[k] ?? k}</td><td>{Math.max(...rows.map((r) => g(r).by_lock_state[k]?.n_doors ?? 0))}</td>
                    {rows.map((r) => { const x = g(r).by_lock_state[k]; return <td key={r.file}>{x ? `${x.doors_solved} (${pct(x.success_rate, 0)})` : "–"}</td>; })}
                  </tr>
                ))}
                {["1", "2", "3", "4", "5"].filter((k) => rows.some((r) => g(r).by_difficulty[k])).map((k) => (
                  <tr key={k}><td>difficulty L{k}</td><td>{Math.max(...rows.map((r) => g(r).by_difficulty[k]?.n_doors ?? 0))}</td>
                    {rows.map((r) => { const x = g(r).by_difficulty[k]; return <td key={r.file}>{x ? `${x.doors_solved} (${pct(x.success_rate, 0)})` : "–"}</td>; })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {current && cur && (
        <div className="rsection">
          <h2>Per door <span className="sub">{current.policy.replace(/_/g, " ")} · {suite} suite · {cur.scenarios.join(", ")} · {current.seeds.length} seed(s) · click a cell to open the door</span></h2>
          <div className="rtabs">
            {rows.map((r) => <button key={r.file} className={r.file === current.file ? "primary" : ""} onClick={() => setSel(r.file)}>{r.policy.replace(/_/g, " ")}</button>)}
            <span className="legend"><i className="ok" /> every episode <i className="part" /> some episodes <i className="bad" /> none <i className="none" /> not evaluated</span>
          </div>
          <div className="doorgrid">
            {doors.map((d) => {
              const s = cur.doors[d.id];
              const scen = suite === "core" ? d.benchmark?.core ?? [] : d.benchmark?.human ?? [];
              return <a key={d.id} href={`#/door/${d.id}`} className={`cell ${outcomeClass(s)}`} aria-label={`${d.id}: ${s ? `${s[0]} of ${s[1]} episodes successful` : "not evaluated"}`} title={`${d.id} · ${FAMILY_LABELS[d.family] ?? d.family} · ${scen.join(", ")} · L${d.difficulty}${d.lock_engaged ? " · locked" : ""}\n${s ? `${s[0]} / ${s[1]} episodes successful` : "not evaluated"}`} />;
            })}
          </div>
          <div className="rmeta">
            <span>{cur.n_episodes} episodes · {cur.outcomes && Object.entries(cur.outcomes).map(([k, v]) => `${v} ${k}`).join(" · ")}{cur.n_errors ? ` · ${cur.n_errors} errors` : ""}</span>
            <span>time budget {typeof current.time_budget_s === "number" ? `${current.time_budget_s} s (override)` : "each scenario's own"} · {current.randomize ? "domain randomisation on seeds ≥ 1" : "nominal physics"} · {cur.mean_wall_s !== null && cur.mean_wall_s !== undefined ? `${cur.mean_wall_s.toFixed(2)} s wall / episode` : ""}{current.wall_time_s ? ` · ${(current.wall_time_s / 60).toFixed(1)} min total` : ""}</span>
            {current.host && <span>{current.host}{current.label ? ` · ${current.label}` : ""}</span>}
            {current.extra && Object.keys(current.extra).length > 0 && <span>{Object.entries(current.extra).map(([k, v]) => `${k}: ${String(v).slice(0, 40)}`).join(" · ")}</span>}
            <span>{current.policy_class}</span>
          </div>
        </div>
      )}
      <div className="about" style={{ paddingTop: 0 }}>
        <p style={{ color: "var(--muted)", fontSize: 12 }}>Index generated {idx.generated.slice(0, 10)} by <code>scripts/build_results_index.py</code>. Reproduce any row with <code>doorbench benchmark run --policy &lt;name&gt; --doors all --seeds 3</code> (core suite) or <code>--suite human</code>.</p>
      </div>
    </div>
  );
}
