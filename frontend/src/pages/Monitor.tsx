import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Pause, Play, RefreshCw, SkipForward } from 'lucide-react'
import { Callout, Card, Empty, PageHead, RunPicker, Slider, Toggle } from '../components/ui'
import Spectrum, { SpectrumLegend } from '../components/Spectrum'
import { api, COLORS, fmt, pct } from '../lib/api'
import { useApp, useToast } from '../lib/hooks'

const POL = ['ql', 'greedy', 'random'] as const
const TITLES: Record<string, string> = { ql: 'Q-learning (trained policy, ε = 0)', greedy: 'Greedy — best sensed-free channel by SINR', random: 'Random — uniformly random channel' }

function counters(sim: any, upto: number) {
  let ok = 0, pu = 0, su = 0, hov = 0
  const N = sim.N
  for (let t = 0; t < Math.min(upto, sim.slots); t++) for (let i = 0; i < N; i++) {
    const c = sim.code[t][i]; if (c === 1) ok++; else if (c === 2) pu++; else if (c === 3) su++
    if (t > 0 && sim.act[t][i] > 0 && sim.act[t - 1][i] > 0 && sim.act[t][i] !== sim.act[t - 1][i]) hov++
  }
  return { ok, pu, su, hov }
}

export default function Monitor() {
  const { runs, runId, setRunId } = useApp()
  const toast = useToast()
  const [slots, setSlots] = useState(120), [seed, setSeed] = useState(2025), [ov, setOv] = useState(false), [eta, setEta] = useState(0.4)
  const [sims, setSims] = useState<Record<string, any> | null>(null)
  const [loading, setLoading] = useState(false)
  const [t, setT] = useState(0), [playing, setPlaying] = useState(false), [speed, setSpeed] = useState(12)
  const load = async (s = seed) => {
    if (!runId) return
    setLoading(true); setPlaying(false)
    try {
      const out = await Promise.all(POL.map(p => api(`/api/runs/${runId}/simulate`, { body: { policy: p, slots, seed: s, eta_pu: ov ? eta : null } })))
      setSims(Object.fromEntries(POL.map((p, i) => [p, out[i]]))); setT(0); setPlaying(true)
    } catch (e: any) { toast.push(e.message, 'err') } finally { setLoading(false) }
  }
  useEffect(() => { setSims(null); if (runId) load() }, [runId])
  const total = sims?.ql.slots || slots
  useEffect(() => {
    if (!playing || !sims) return
    const id = setInterval(() => setT(x => { if (x >= total) { setPlaying(false); return x } return x + 1 }), 1000 / speed)
    return () => clearInterval(id)
  }, [playing, speed, total, sims])
  const cnt = useMemo(() => sims ? Object.fromEntries(POL.map(p => [p, counters(sims[p], t)])) : null, [sims, t])
  const reroll = () => { const s = Math.floor(Math.random() * 100000); setSeed(s); load(s) }
  const N = sims?.ql.N
  return <div className="stack">
    <PageHead kicker="Live spectrum monitor" title={<>Same radio, three <em className="hl">decision makers</em></>}
      sub="Each panel replays the identical PU activity and fading realisation; only the allocation policy differs. Blue = primary users, amber = successful SU transmissions, red = PU collisions, violet = SU–SU collisions."
      right={<RunPicker runs={runs} value={runId} onChange={setRunId} />} />
    {!runId ? <Card><Empty title="No trained policy yet">Train an agent in the <Link to="/lab/train" style={{ color: 'var(--ql)' }}>Training studio</Link> and it will appear here.</Empty></Card> : <>
      <Card>
        <div className="grid g4" style={{ alignItems: 'end' }}>
          <Slider label="Slots to simulate" value={slots} min={40} max={300} step={10} onChange={setSlots} />
          <div className="field"><label><span>Seed</span></label><input type="number" value={seed} onChange={e => setSeed(+e.target.value)} /></div>
          <div><Toggle on={ov} onChange={setOv} label="Override PU activity η" />{ov && <Slider label="η_PU" value={eta} min={0.05} max={0.95} step={0.05} onChange={setEta} fmt={v => v.toFixed(2)} />}</div>
          <div className="row" style={{ marginBottom: 14 }}><button className="btn primary" disabled={loading} onClick={() => load()}><RefreshCw size={15} />Simulate</button><button className="btn" disabled={loading} onClick={reroll}>New seed</button></div>
        </div>
      </Card>
      {!sims ? <Card><div className="empty">{loading ? 'Running the simulator…' : 'Press Simulate'}</div></Card> : <>
        <Card>
          <div className="row">
            <button className="btn sm primary" onClick={() => { if (t >= total) setT(0); setPlaying(!playing) }} aria-label={playing ? 'Pause' : 'Play'}>{playing ? <Pause size={14} /> : <Play size={14} />}{playing ? 'Pause' : 'Play'}</button>
            <button className="btn sm" onClick={() => { setPlaying(false); setT(Math.min(total, t + 1)) }} aria-label="Step"><SkipForward size={14} />Step</button>
            <div style={{ flex: 1, minWidth: 200 }}><input type="range" min={0} max={total} value={t} onChange={e => { setPlaying(false); setT(+e.target.value) }} style={{ ['--p' as any]: `${t / total * 100}%` }} aria-label="Time slot" /></div>
            <span className="mono note">slot {t}/{total}</span>
            <div style={{ width: 150 }}><Slider label="Speed" value={speed} min={2} max={40} step={1} onChange={setSpeed} fmt={v => `${v}/s`} /></div></div>
          <SpectrumLegend />
        </Card>
        {POL.map(p => <Card key={p} kicker={`${N} SUs · ${sims[p].K} channels`} title={<span style={{ color: (COLORS as any)[p] }}>{TITLES[p]}</span>}
          right={cnt && <div className="row" style={{ gap: 8 }}><span className="pill ok">✓ {cnt[p].ok}</span><span className="pill bad">PU hit {cnt[p].pu}</span><span className="pill" style={{ borderColor: '#B26BFF66', color: '#d4b4ff' }}>SU hit {cnt[p].su}</span><span className="pill warn">handovers {cnt[p].hov}</span></div>}>
          <Spectrum occ={sims[p].occ} act={sims[p].act} code={sims[p].code} K={sims[p].K} upto={t} height={270} label={`${p} policy spectrum waterfall`} />
          <div className="row" style={{ marginTop: 10, gap: 22 }}>
            {[['Throughput', `${fmt(sims[p].throughput_mbps, 0)} Mbps`], ['PU collisions', pct(sims[p].pu_collision)], ['SU collisions', pct(sims[p].su_collision)], ['Utilisation', pct(sims[p].utilisation)], ['Handovers', `${fmt(sims[p].handovers_per_min, 1)}/min`]].map(([l, v]) =>
              <span key={l} className="note"><span className="faint">{l} (whole trace)</span> <b className="mono" style={{ color: '#E8ECF6' }}>{v}</b></span>)}</div>
        </Card>)}
        <Callout kind="info">A single {total}-slot trace is a vivid illustration, not a statistic — it is one random draw. For averages over many seeds see <Link to="/results/benchmarks" style={{ color: 'var(--ql)' }}>Benchmarks</Link>. Counters above are computed from the traced decisions; "whole trace" metrics come from the simulator.</Callout>
      </>}
    </>}
  </div>
}
