import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, FlaskConical, Gauge } from 'lucide-react'
import { Card, PageHead, Stat, Skeleton } from '../components/ui'
import Spectrum, { SpectrumLegend } from '../components/Spectrum'
import { api, fmt } from '../lib/api'
import { useApp, useAsync } from '../lib/hooks'

function Hero({ runId }: { runId?: number }) {
  const sim = useAsync(() => api(`/api/runs/${runId}/simulate`, { body: { policy: 'ql', slots: 90, seed: 2024 } }), [runId], !!runId)
  const [t, setT] = useState(0)
  const timer = useRef<any>()
  useEffect(() => { if (!sim.data) return; clearInterval(timer.current); timer.current = setInterval(() => setT(x => (x + 1) % (sim.data.slots + 25)), 120); return () => clearInterval(timer.current) }, [sim.data])
  return <div className="hero">
    <div className="content">
      <div className="mono" style={{ color: 'var(--pu)', fontSize: 11.5, letterSpacing: '.14em' }}>REINFORCEMENT LEARNING · COGNITIVE RADIO NETWORKS</div>
      <h1>The spectrum is full on paper — <em className="hl">idle in practice.</em></h1>
      <p>SpectraRL trains tabular Q-learning agents that sense primary-user activity, choose a channel every time slot, and balance throughput, PU protection and spectrum switching — then benchmarks them against greedy and random allocation in a Rayleigh-fading CRN simulator.</p>
      <div className="row" style={{ marginTop: 18 }}>
        <Link to="/lab/train" className="btn primary"><FlaskConical size={16} />Train an agent</Link>
        <Link to="/results/benchmarks" className="btn"><Gauge size={16} />See measured results</Link>
      </div>
    </div>
    {sim.data && <div style={{ position: 'absolute', right: 0, top: 0, height: '100%', width: '46%', pointerEvents: 'none', maskImage: 'linear-gradient(90deg,transparent,#000 35%)', WebkitMaskImage: 'linear-gradient(90deg,transparent,#000 35%)' }}>
      <Spectrum mini occ={sim.data.occ} act={sim.data.act} code={sim.data.code} K={sim.data.K} upto={Math.min(t, sim.data.slots)} height={420} label="Live trace of the trained agent" /></div>}
  </div>
}

function Kpi({ label, measured, paper, better, sub }: { label: string; measured: string; paper: string; better?: boolean | null; sub?: string }) {
  return <Stat label={label} value={measured} color={better == null ? undefined : better ? '#7ee8bf' : '#fca5a5'}
    sub={<><span className={`pill ${better == null ? '' : better ? 'ok' : 'bad'}`}>{better == null ? 'measured' : better ? 'Q-learning better' : 'Q-learning worse'}</span><span className="faint">paper: {paper}</span>{sub && <span className="faint">{sub}</span>}</>} />
}

export default function Overview() {
  const { runId, meta } = useApp()
  const res = useAsync(() => api('/api/experiments/results/latest'), [])
  const R = res.data?.results || {}
  const P = meta?.paper
  const thr = R.throughput_vs_n, col = R.collision_vs_eta, ut = R.utilisation, ab = R.ablation, cv = R.convergence, rt = R.runtime
  const have = !!(thr && col && ut && ab && cv && rt)
  const iN = thr ? thr.N.indexOf(40) : -1, iE = col ? col.eta.indexOf(0.4) : -1
  const gain = have ? thr.ql.throughput_mbps[iN] / thr.greedy.throughput_mbps[iN] - 1 : 0
  const cred = have ? 1 - col.ql.pu_collision[iE] / col.greedy.pu_collision[iE] : 0
  const dut = have ? (ut.ql.avg - ut.greedy.avg) * 100 : 0
  const ql0 = ab?.variants?.[0], gr = ab?.variants?.find((v: any) => v.name === 'Greedy')
  const hred = have ? 1 - ql0.handovers_per_min / gr.handovers_per_min : 0
  const ep90 = have ? cv.episodes_to_90.reduce((a: number, b: number) => a + b, 0) / cv.episodes_to_90.length : 0
  return <div>
    <Hero runId={runId} />
    <div className="row" style={{ marginBottom: 14 }}><SpectrumLegend /><span className="sp" /><span className="note">Trace above: the trained agent (amber) hopping between white-space gaps of the PU pattern (blue) — real simulator output.</span></div>
    <div className="page-head" style={{ marginTop: 26, marginBottom: 12 }}><div><div className="kick">Headline results</div><h2 style={{ fontSize: 22 }}>Measured here vs. claimed in the presentation</h2></div>
      <Link to="/results/benchmarks" className="btn sm">All experiments <ArrowRight size={14} /></Link></div>
    {res.loading ? <Skeleton h={150} /> : !have ? <Card><p>No benchmark results stored yet. Run the full experiment suite from <Link to="/results/benchmarks" style={{ color: 'var(--ql)' }}>Benchmarks</Link> to populate this panel with measured numbers.</p></Card> :
      <div className="grid g3">
        <Kpi label={`Throughput gain vs greedy (N=40)`} measured={`${gain >= 0 ? '+' : ''}${fmt(gain * 100, 1)}%`} paper="+62.5%" better={gain > 0} sub={`${fmt(thr.ql.throughput_mbps[iN], 0)} vs ${fmt(thr.greedy.throughput_mbps[iN], 0)} Mbps`} />
        <Kpi label="PU-collision reduction vs greedy (η=0.4)" measured={`${cred >= 0 ? '' : '+'}${fmt(Math.abs(cred) * 100, 1)}%${cred >= 0 ? ' fewer' : ' more'}`} paper="73.2% fewer" better={cred > 0} sub={`${fmt(col.ql.pu_collision[iE] * 100, 1)}% vs ${fmt(col.greedy.pu_collision[iE] * 100, 1)}%`} />
        <Kpi label="Spectrum utilisation vs greedy" measured={`${dut >= 0 ? '+' : ''}${fmt(dut, 1)} pts`} paper="+23.7 pts" better={dut > 0} sub={`${fmt(ut.ql.avg * 100, 1)}% vs ${fmt(ut.greedy.avg * 100, 1)}%`} />
        <Kpi label="Handover reduction vs greedy" measured={`${fmt(hred * 100, 1)}% fewer`} paper="71.2% fewer" better={hred > 0} sub={`${fmt(ql0.handovers_per_min, 1)} vs ${fmt(gr.handovers_per_min, 1)} /min`} />
        <Kpi label="Episodes to reach 90% of final reward" measured={fmt(ep90, 0)} paper="≈320–350" better={null} sub="mean of 3 seeds" />
        <Kpi label="Decision latency (Python inference path)" measured={`${fmt(rt.python_ms.ql, 4)} ms`} paper="0.42 ms" better={null} sub="different hardware / implementation" />
      </div>}
    <div className="grid g3" style={{ marginTop: 20 }}>
      {[['1. Understand', 'Walk through the CRN system model, the MDP, the multi-objective reward and Algorithm 1 — each page runs the real code, not slides.', '/system'],
        ['2. Train', 'Configure the network and hyperparameters, train live, and watch Q-learning overtake greedy and random episode by episode.', '/train'],
        ['3. Inspect & verify', 'Replay decisions on the spectrum waterfall, query the Q-table, and regenerate every results slide from fresh experiments.', '/monitor']].map(([t, d, to]) =>
        <Link key={t} to={to} className="card" style={{ display: 'block' }}><h2>{t}</h2><p className="muted" style={{ marginTop: 8 }}>{d}</p><span className="note">Open <ArrowRight size={12} style={{ verticalAlign: -2 }} /></span></Link>)}
    </div>
  </div>
}
