import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Download, Play, Square, Trash2, ChevronDown } from 'lucide-react'
import { Callout, Card, Empty, Legend, NumField, PageHead, RunPicker, Slider, StatusPill, Stat, Tabs } from '../components/ui'
import { LineChartX } from '../components/charts'
import { api, COLORS, fmt, pct, smooth } from '../lib/api'
import { useApp, usePoll, useToast } from '../lib/hooks'

const PRESETS: Record<string, { label: string; env?: any; ql?: any }> = {
  paper: { label: 'Paper defaults (1000 × 500)', env: {}, ql: { episodes: 1000, slots: 500 } },
  quick: { label: 'Quick demo (300 × 200)', env: {}, ql: { episodes: 300, slots: 200 } },
  heavy: { label: 'Heavy load (N = 30)', env: { N: 30 }, ql: { episodes: 1000, slots: 500 } },
  strict: { label: 'Strict paper reward (ρ_su = 0)', env: {}, ql: { rho_su_collision: 0, episodes: 1000, slots: 500 } },
  protect: { label: 'PU-protection (ρ_coll = 5)', env: {}, ql: { rho_collision: 5, episodes: 1000, slots: 500 } },
}
type MetricKey = 'reward' | 'throughput_mbps' | 'pu_collision' | 'su_collision' | 'utilisation' | 'handovers_per_min'
const METRICS: { id: MetricKey; label: string; fmt: (v: number) => string }[] = [
  { id: 'reward', label: 'Reward / episode', fmt: v => fmt(v, 0) }, { id: 'throughput_mbps', label: 'Throughput (Mbps)', fmt: v => fmt(v, 0) },
  { id: 'pu_collision', label: 'PU collision', fmt: v => pct(v, 0) }, { id: 'su_collision', label: 'SU collision', fmt: v => pct(v, 0) },
  { id: 'utilisation', label: 'Utilisation', fmt: v => pct(v, 0) }, { id: 'handovers_per_min', label: 'Handovers / min', fmt: v => fmt(v, 0) }]

function Section({ title, children, open: o = false }: { title: string; children: any; open?: boolean }) {
  const [open, setOpen] = useState(o)
  return <div style={{ borderTop: '1px solid var(--border)', paddingTop: 10, marginTop: 10 }}>
    <button className="row" style={{ background: 'none', border: 0, width: '100%', cursor: 'pointer', padding: 0, marginBottom: open ? 12 : 0 }} onClick={() => setOpen(!open)} aria-expanded={open}>
      <b style={{ fontSize: 13 }}>{title}</b><span className="sp" /><ChevronDown size={16} style={{ transform: open ? 'rotate(180deg)' : undefined, transition: '.2s' }} /></button>{open && children}</div>
}

export default function Train() {
  const { meta, runs, refreshRuns, setRunId, runId } = useApp()
  const toast = useToast()
  const [env, setEnv] = useState<any>(null), [ql, setQl] = useState<any>(null)
  const [name, setName] = useState('My training run'), [baselines, setBaselines] = useState(true)
  useEffect(() => { if (meta && !env) { setEnv({ ...meta.defaults.env }); setQl({ ...meta.defaults.ql }) } }, [meta])
  const [active, setActive] = useState<number | undefined>(() => { const v = sessionStorage.getItem('train_active'); return v ? +v : undefined })
  const [metric, setMetric] = useState<MetricKey>('reward')
  const [busy, setBusy] = useState(false)
  const rows = useRef<any[]>([]); const [, force] = useState(0)
  const lastEp = useRef(0)
  useEffect(() => { rows.current = []; lastEp.current = 0; force(x => x + 1); if (active) sessionStorage.setItem('train_active', String(active)) }, [active])
  const run = usePoll(() => api(`/api/runs/${active}`), 700, [active], !!active)
  const live = run.data
  const running = live && (live.status === 'running' || live.status === 'queued')
  const mq = usePoll(async () => {
    const m: any[] = await api(`/api/runs/${active}/metrics?since=${lastEp.current}`)
    if (m.length) { rows.current = rows.current.concat(m); lastEp.current = m[m.length - 1].episode; force(x => x + 1) }
    return m.length
  }, 700, [active, live?.status], !!active)
  useEffect(() => { if (live && !running) refreshRuns() }, [live?.status])

  const series = useMemo(() => {
    const by: Record<string, any[]> = { ql: [], greedy: [], random: [] }
    rows.current.forEach(r => by[r.policy].push(r))
    const eps = by.ql.map(r => r.episode)
    const sm: Record<string, number[]> = {}
    for (const p of ['ql', 'greedy', 'random']) sm[p] = smooth(by[p].map(r => r[metric]), 10)
    return { eps, data: eps.map((e, i) => ({ episode: e, ql: sm.ql[i], greedy: sm.greedy[i], random: sm.random[i], eps: by.ql[i]?.eps })), by }
  }, [rows.current.length, metric])
  const M = METRICS.find(m => m.id === metric)!
  const total = live?.episodes_total || 0, done = live?.episodes_done || 0
  const last = series.data[series.data.length - 1]
  const elapsed = live ? ((live.finished_at || Date.now() / 1000) - live.created_at) : 0
  const eta = running && done > 5 ? elapsed / done * (total - done) : 0

  const preset = (k: string) => { const p = PRESETS[k]; setEnv({ ...meta.defaults.env, ...p.env }); setQl({ ...meta.defaults.ql, ...p.ql }) }
  const start = async () => {
    setBusy(true)
    try { const r: any = await api('/api/runs', { body: { name, env, ql, baselines } }); setActive(r.id); setRunId(r.id); refreshRuns(); toast.push(`Training run #${r.id} started`, 'ok') }
    catch (e: any) { toast.push(e.message, 'err') } finally { setBusy(false) }
  }
  const stop = async () => { try { await api(`/api/runs/${active}/stop`, { method: 'POST', body: {} }); toast.push('Stopping after the current episode…') } catch (e: any) { toast.push(e.message, 'err') } }
  const del = async () => { if (!confirm(`Delete run #${active} and its policy?`)) return; try { await api(`/api/runs/${active}`, { method: 'DELETE' }); setActive(undefined); sessionStorage.removeItem('train_active'); refreshRuns(); toast.push('Run deleted', 'ok') } catch (e: any) { toast.push(e.message, 'err') } }
  if (!env || !ql) return <div className="empty">Loading defaults…</div>
  const E = (k: string) => (v: number) => setEnv((s: any) => ({ ...s, [k]: v })), Q = (k: string) => (v: number) => setQl((s: any) => ({ ...s, [k]: v }))
  const summary = live?.summary_json ? JSON.parse(live.summary_json) : null
  const cfgEnv = live ? JSON.parse(live.env_json) : null

  return <div className="stack">
    <PageHead kicker="Lab · step 1" title={<>Train a Q-learning agent <em className="hl">live</em></>}
      sub="Every run trains real tabular Q-learning agents (one table per SU) in the compiled CRN simulator, alongside greedy and random allocators on identical radio conditions. Nothing here is simulated for display." />
    <div className="grid g-side" style={{ gridTemplateColumns: '350px minmax(0,1fr)', alignItems: 'start' }}>
      <Card kicker="Configuration" title="New training run" className="">
        <div className="field"><label><span>Presets</span></label><select defaultValue="" onChange={e => e.target.value && preset(e.target.value)} aria-label="Preset"><option value="" disabled>Choose a preset…</option>{Object.entries(PRESETS).map(([k, p]) => <option key={k} value={k}>{p.label}</option>)}</select></div>
        <div className="field"><label><span>Run name</span></label><input type="text" value={name} maxLength={80} onChange={e => setName(e.target.value)} /></div>
        <Slider label={<>Secondary users N</>} value={env.N} min={1} max={64} step={1} onChange={E('N')} />
        <Slider label={<>PU activity η<sub>PU</sub></>} value={env.eta_pu} min={0.05} max={0.95} step={0.05} onChange={E('eta_pu')} fmt={v => v.toFixed(2)} />
        <Slider label={<>Learning rate α</>} value={ql.alpha} min={0.01} max={0.5} step={0.01} onChange={Q('alpha')} fmt={v => v.toFixed(2)} />
        <div className="grid g2" style={{ columnGap: 12 }}><NumField label="Episodes" value={ql.episodes} min={1} max={20000} step={50} onChange={Q('episodes')} /><NumField label="Slots / episode" value={ql.slots} min={10} max={5000} step={50} onChange={Q('slots')} /></div>
        <Section title="Network (Table 3)">
          <div className="grid g2" style={{ columnGap: 12 }}><NumField label="Channels K" value={env.K} min={2} max={32} step={1} onChange={E('K')} /><NumField label="PUs M" value={env.M} min={1} max={32} step={1} onChange={E('M')} />
            <NumField label="P_tx (dBm)" value={env.ptx_dbm} step={1} onChange={E('ptx_dbm')} /><NumField label="Path-loss α" value={env.path_loss_exp} step={0.1} onChange={E('path_loss_exp')} /></div>
          <Slider label="PU dynamics (speed)" value={env.pu_dynamics} min={0.05} max={4} step={0.05} onChange={E('pu_dynamics')} fmt={v => `×${v.toFixed(2)}`} />
          <Slider label="Packet arrival prob." value={env.arrival_prob} min={0} max={1} step={0.05} onChange={E('arrival_prob')} fmt={v => v.toFixed(2)} />
          <Slider label="Sensing false-alarm" value={env.p_false_alarm} min={0} max={0.3} step={0.01} onChange={E('p_false_alarm')} fmt={v => v.toFixed(2)} />
          <Slider label="Sensing miss-detect" value={env.p_miss_detect} min={0} max={0.3} step={0.01} onChange={E('p_miss_detect')} fmt={v => v.toFixed(2)} />
          <Slider label="Switch overhead (slot fraction)" value={env.switch_overhead} min={0} max={0.6} step={0.05} onChange={E('switch_overhead')} fmt={v => v.toFixed(2)} />
          <div className="field"><label><span>SU contention model</span></label><select value={env.su_contention} onChange={e => setEnv({ ...env, su_contention: e.target.value })}><option value="domain">single collision domain</option><option value="sinr">SINR threshold (spatial reuse)</option></select></div>
        </Section>
        <Section title="Q-learning (Algorithm 1)">
          <div className="grid g2" style={{ columnGap: 12 }}><NumField label="γ" value={ql.gamma} min={0} max={1} step={0.01} onChange={Q('gamma')} /><NumField label="ε₀" value={ql.eps0} min={0} max={1} step={0.05} onChange={Q('eps0')} />
            <NumField label="ε_min" value={ql.eps_min} min={0} max={1} step={0.01} onChange={Q('eps_min')} /><NumField label="δ decay" value={ql.eps_decay} min={0.5} max={1} step={0.001} onChange={Q('eps_decay')} /><NumField label="Seed" value={ql.seed} step={1} onChange={Q('seed')} /></div>
        </Section>
        <Section title="Reward weights (Eq. 3)">
          <div className="grid g2" style={{ columnGap: 12 }}><NumField label="w₁ throughput" value={ql.w_throughput} min={0} step={0.1} onChange={Q('w_throughput')} /><NumField label="ρ_coll (PU)" value={ql.rho_collision} min={0} step={0.25} onChange={Q('rho_collision')} />
            <NumField label="ρ_switch" value={ql.rho_switch} min={0} step={0.05} onChange={Q('rho_switch')} /><NumField label="ρ_su (SU coll.)" value={ql.rho_su_collision} min={0} step={0.1} onChange={Q('rho_su_collision')} />
            <NumField label="idle bonus" value={ql.idle_reward} min={0} step={0.05} onChange={Q('idle_reward')} /><NumField label="idle penalty" value={ql.idle_penalty} min={0} step={0.05} onChange={Q('idle_penalty')} /></div>
        </Section>
        <div className="row" style={{ marginTop: 16 }}>
          <button className="btn primary" disabled={busy} onClick={start} style={{ flex: 1, justifyContent: 'center' }}><Play size={16} />Train</button>
          <label className="toggle" style={{ fontSize: 12.5 }}><input type="checkbox" checked={baselines} onChange={e => setBaselines(e.target.checked)} /> run baselines</label></div>
        <p className="note" style={{ marginTop: 10 }}>Rewards: ρ_su is an <b>extension</b> to the deck's Eq. (3) (set 0 for the strict form). Weights are not published in the deck.</p>
      </Card>

      <div className="stack">
        <Card kicker="Run" title={live ? `#${live.id} · ${live.name}` : 'No run selected'} right={<div className="row">
          <RunPicker runs={runs} value={active} onChange={setActive} onlyReady={false} />
          {running && <button className="btn sm danger" onClick={stop}><Square size={13} />Stop</button>}
          {live && !running && <><Link className="btn sm" to="/lab/monitor" onClick={() => setRunId(live.id)}>Open in monitor</Link><a className="btn sm" href={`/api/runs/${live.id}/policy`}><Download size={13} />Policy</a><button className="btn sm" onClick={del} aria-label="Delete run"><Trash2 size={13} /></button></>}</div>}>
          {!live ? <Empty title="Configure and press Train">Curves appear here episode by episode. Pick a past run above to review it.</Empty> : <>
            <div className="row" style={{ marginBottom: 8 }}><StatusPill s={live.status} /><span className="note">episode {done} / {total}</span><span className="sp" />
              <span className="note mono">ε = {fmt(last?.eps ?? 1, 3)} · {fmt(elapsed, 0)} s{running && eta ? ` · ETA ${fmt(eta, 0)} s` : ''}</span></div>
            <div className="bar"><div style={{ width: `${total ? done / total * 100 : 0}%` }} /></div>
            {live.error && <Callout kind="bad">{live.error}</Callout>}
          </>}
        </Card>
        {live && <div className="grid g4">
          <Stat label="Q-learning reward (10-ep avg)" value={last ? fmt(last.ql, 1) : '–'} color={COLORS.ql} />
          <Stat label="Greedy" value={last && baselines ? fmt(series.data.length ? smooth(series.by.greedy.map(r => r.reward), 10).slice(-1)[0] : 0, 1) : '–'} color={COLORS.greedy} />
          <Stat label="Random" value={last && baselines ? fmt(smooth(series.by.random.map(r => r.reward), 10).slice(-1)[0], 1) : '–'} color={COLORS.random} />
          <Stat label="Exploration ε" value={fmt(last?.eps ?? 1, 3)} sub={<span className="faint">γ = {ql.gamma}</span>} /></div>}
        <Card kicker="Live metrics" title={M.label} right={<Tabs tabs={METRICS.map(m => ({ id: m.id, label: m.label.split(' ')[0] === 'Reward' ? 'Reward' : m.label.replace(' / min', '').replace(' (Mbps)', '') }))} value={metric} onChange={setMetric} />}>
          <Legend items={[{ name: 'Q-learning', color: COLORS.ql }, { name: 'Greedy', color: COLORS.greedy }, { name: 'Random', color: COLORS.random }]} />
          {series.data.length ? <LineChartX data={series.data} xKey="episode" height={330} xLabel="training episode" yFmt={M.fmt} series={[{ key: 'ql', name: 'Q-learning', color: COLORS.ql, width: 3 }, { key: 'greedy', name: 'Greedy', color: COLORS.greedy }, { key: 'random', name: 'Random', color: COLORS.random }]} />
            : <Empty title="Waiting for the first episode…" />}
          <p className="note">Smoothed over 10 episodes. Greedy and random never learn, so their curves are flat apart from sampling noise — they run on exactly the same radio conditions as the learner each episode.</p>
        </Card>
        {summary && !running && <Card kicker="Result" title="Last-20-episode averages">
          <div className="scroll-x"><table className="t"><thead><tr><th>Policy</th><th className="n">Reward</th><th className="n">Throughput</th><th className="n">PU coll.</th><th className="n">SU coll.</th><th className="n">Utilisation</th><th className="n">Handovers/min</th></tr></thead><tbody>
            {Object.entries(summary).map(([p, m]: any) => <tr key={p}><td><span style={{ color: (COLORS as any)[p] }}>●</span> {p === 'ql' ? 'Q-learning' : p}</td><td className="n">{fmt(m.reward, 1)}</td><td className="n">{fmt(m.throughput_mbps, 0)} Mbps</td><td className="n">{pct(m.pu_collision)}</td><td className="n">{pct(m.su_collision)}</td><td className="n">{pct(m.utilisation)}</td><td className="n">{fmt(m.handovers_per_min, 1)}</td></tr>)}</tbody></table></div>
          <p className="note" style={{ marginTop: 8 }}>These are online training-time averages (the learner still explores). Frozen-policy evaluation on held-out seeds is on the <Link to="/results/benchmarks" style={{ color: 'var(--ql)' }}>Benchmarks</Link> page and in the monitor.</p></Card>}
      </div>
    </div>
  </div>
}
