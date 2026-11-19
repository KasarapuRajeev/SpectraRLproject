import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Bar, BarChart, ResponsiveContainer, XAxis, YAxis, Tooltip, Cell } from 'recharts'
import { Callout, Card, PageHead, Slider, Tabs, Toggle } from '../components/ui'
import { api, fmt } from '../lib/api'
import { useApp, useAsync } from '../lib/hooks'

const BLOCKS: any[] = [
  [30, 0, 'Primary users', 'Markov ON/OFF traffic', '#38BDF8', 'Licensed users with full priority. Each PU is a two-state Markov source with stationary ON probability η_PU and its own switching speed; it owns channels j, j+M, …', 'core.run_episode · PU transitions'],
  [30, 1, 'Multi-channel spectrum', 'K = 16, Rayleigh fading', '#38BDF8', 'Sixteen orthogonal 10 MHz sub-channels with Rayleigh fading (|h|² ~ Exp(1)) and path loss exponent α = 3.8.', 'world.py · build_world'],
  [30, 2, 'SU buffer pool', 'Queues & QoS demand', '#38BDF8', 'Secondary users with Bernoulli packet arrivals into finite queues. An SU with an empty queue stays silent. Queue size is part of the state.', 'queue_bucket() in core.py'],
  [30, 3, 'SINR estimator', 'Interference metrics', '#38BDF8', 'Measures SINR (Eq. 1) incl. co-channel SU interference and active-PU interference, giving capacity (Eq. 2). Try it below.', 'world.sinr · /api/channel'],
  [710, 0, 'State engine  Sₜ', 'PU status, queues, SINR', '#F5A524', 'Builds Sₜ = (C_{t−1}, sensed O[C_{t−1}], best sensed-free channel by SINR, queue bucket, free-channel bucket) → one of 5 202 table rows.', 'core.encode_state'],
  [710, 1, 'ε-greedy selector  aₜ', 'Explore vs exploit', '#F5A524', 'Random action with probability ε, otherwise argmax Q(S,·) with random tie-breaking. ε decays ×0.995 per episode from 1.0 to 0.01.', 'core.argmax_tiebreak'],
  [710, 2, 'Multi-objective reward  Rₜ', 'Thr − Int − Switch', '#F5A524', 'R = w₁T − ρ_coll·I_coll − ρ_sw·H + P_idle (plus an optional SU-collision penalty, off in strict-paper mode).', 'core.reward_terms'],
  [710, 3, 'Bellman Q-updater', 'Q ← Q + α[ΔQ]', '#F5A524', 'Temporal-difference update with α = 0.1, γ = 0.95, executed one slot after the action once the next state is observed.', 'run_episode · Bellman update'],
]

function Arch() {
  const [sel, setSel] = useState(4)
  const b = BLOCKS[sel]
  return <div className="grid g-main">
    <svg viewBox="0 0 1160 640" className="arch" role="img" aria-label="RL-based dynamic spectrum allocation framework" style={{ width: '100%', background: 'rgba(255,255,255,.02)', borderRadius: 14, border: '1px solid var(--border)' }}>
      <defs>{[['ar', '#F87171'], ['ag', '#34D399']].map(([id, c]) => <marker key={id} id={id} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill={c} /></marker>)}</defs>
      {[[30, 'Radio Environment (CRN)', '#38BDF8'], [710, 'Q-Learning Agent', '#F5A524']].map(([x, t, c]: any) => <g key={t}><rect x={x} y={20} width={420} height={600} rx={16} fill={c} fillOpacity={.06} stroke={c} strokeWidth={2} /><text x={x + 210} y={70} textAnchor="middle" fontSize={26} fontWeight={600} fill={c}>{t}</text></g>)}
      {BLOCKS.map((bk, i) => { const [x, r, t, s, c] = bk, y = 105 + r * 125
        return <g key={i} className={`item ${sel === i ? 'sel' : ''}`} tabIndex={0} role="button" aria-label={t} onClick={() => setSel(i)} onKeyDown={e => e.key === 'Enter' && setSel(i)}>
          <rect x={x + 22} y={y} width={376} height={100} rx={12} fill="rgba(255,255,255,.05)" stroke={c} strokeWidth={sel === i ? 3 : 1.5} />
          <text x={x + 44} y={y + 42} fontSize={22} fontWeight={600} fill="#E8ECF6">{t}</text><text x={x + 44} y={y + 74} fontSize={17} fill="#8B98B4">{s}</text></g> })}
      <line className="flow" x1={700} y1={285} x2={462} y2={285} stroke="#F87171" strokeWidth={5} markerEnd="url(#ar)" />
      <line className="flow" x1={462} y1={440} x2={700} y2={440} stroke="#34D399" strokeWidth={5} markerEnd="url(#ag)" />
      <text x={581} y={252} textAnchor="middle" fontSize={19} fontWeight={600} fill="#F87171">Action aₜ (channel)</text>
      <text x={581} y={485} textAnchor="middle" fontSize={19} fontWeight={600} fill="#34D399">Sₜ₊₁ , reward Rₜ</text>
    </svg>
    <Card kicker={b[0] < 300 ? 'Environment' : 'Agent'} title={b[2]}>
      <p>{b[5]}</p><span className="pill info mono">{b[6]}</span>
      <div className="divider" /><p className="note">Decentralised CRN with <b>M</b> = 8 PUs, <b>N</b> SUs and <b>K</b> = 16 orthogonal channels of bandwidth <b>B<sub>k</sub></b>. PUs have full priority; SUs exploit white space while keeping PU interference low.</p>
    </Card>
  </div>
}

function Channel() {
  const [d, setD] = useState(50); const [pu, setPu] = useState(false); const [ipu, setIpu] = useState(-95)
  const q = useAsync(() => api('/api/channel', { body: { distance_m: d, pu_active_interference_dbm: pu ? ipu : null } }), [d, pu, ipu])
  const r = q.data
  const rows = useMemo(() => r ? r.hist.map((h: number, i: number) => ({ x: (r.edges[i] + r.edges[i + 1]) / 2, h })) : [], [r])
  return <Card kicker="Eq. (1) + (2) · live from the API" title="SINR and Shannon–Hartley capacity under Rayleigh fading">
    <div className="grid g-side" style={{ gridTemplateColumns: '300px 1fr' }}>
      <div>
        <Slider label="SU link distance d" value={d} min={5} max={300} step={5} onChange={setD} fmt={v => `${v} m`} />
        <div style={{ marginBottom: 12 }}><Toggle on={pu} onChange={setPu} label="Active PU interference I_PU" /></div>
        {pu && <Slider label="I_PU at receiver" value={ipu} min={-120} max={-70} step={1} onChange={setIpu} fmt={v => `${v} dBm`} />}
        {r && <dl className="kv"><dt>Noise (10 MHz)</dt><dd>{fmt(r.noise_dbm, 1)} dBm</dd><dt>Mean SINR</dt><dd>{fmt(r.mean_snr_db, 1)} dB</dd><dt>E[C] (Rayleigh)</dt><dd>{fmt(r.mean_capacity_mbps, 1)} Mbps</dd>
          <dt>Jensen bound</dt><dd>{fmt(r.jensen_bound_mbps, 1)} Mbps</dd><dt>5–95 % range</dt><dd>{fmt(r.p5_mbps, 0)}–{fmt(r.p95_mbps, 0)} Mbps</dd><dt>Outage P(γ&lt;3 dB)</dt><dd>{fmt(r.outage_prob_3db * 100, 1)} %</dd></dl>}
      </div>
      <div style={{ height: 250 }}><ResponsiveContainer><BarChart data={rows}><XAxis dataKey="x" tickFormatter={v => fmt(v, 0)} stroke="#5B6788" fontSize={11} tickLine={false} label={{ value: 'capacity C (Mbps) — distribution over 20 000 fading draws', position: 'insideBottom', offset: -2, fill: '#8B98B4', fontSize: 11 }} height={40} /><YAxis hide />
        <Tooltip contentStyle={{ background: '#0b1224', border: '1px solid rgba(148,163,184,.3)', borderRadius: 10 }} formatter={(v: any) => [v, 'draws']} labelFormatter={v => `${fmt(+v, 1)} Mbps`} />
        <Bar dataKey="h" radius={[3, 3, 0, 0]} isAnimationActive={false}>{rows.map((_: any, i: number) => <Cell key={i} fill="#4C8DFF" fillOpacity={.85} />)}</Bar></BarChart></ResponsiveContainer></div>
    </div>
    <div className="grid g2" style={{ marginTop: 12 }}>
      <div className="eq"><span>γ<sub>i,k</sub>(t) = P<sub>tx</sub>|h<sub>i,k</sub>|² d<sub>i</sub><sup>−α</sup> / (σ<sub>n</sub>² + Σ<sub>j≠i</sub> I<sub>SU</sub> + I<sub>PU,k</sub>)</span><span className="tg">(1)</span></div>
      <div className="eq"><span>C<sub>i,k</sub>(t) = B<sub>k</sub> log₂(1 + γ<sub>i,k</sub>(t))</span><span className="tg">(2)</span></div></div>
  </Card>
}

const MDP: Record<string, [string, JSX.Element]> = {
  S: ['State space', <><div className="eq"><span>S<sub>t</sub> = (O<sub>t</sub>, Q<sub>t</sub>, H<sub>t</sub>, C<sub>t−1</sub>)</span></div><dl className="kv" style={{ marginTop: 12 }}><dt>O<sub>t</sub> — PU occupancy</dt><dd>K entries → compressed to occupancy of C<sub>t−1</sub> + free-channel bucket</dd><dt>Q<sub>t</sub> — SU queue</dt><dd>bucketed {'{'}empty, 1–3, 4+{'}'}</dd><dt>H<sub>t</sub> — channel SINR</dt><dd>index of best sensed-free channel</dd><dt>C<sub>t−1</sub></dt><dd>previous channel (0…K)</dd></dl><p className="note" style={{ marginTop: 10 }}>The full state space is astronomically large (2<sup>16</sup> occupancy patterns alone). "State compression" yields <b>17 × 2 × 17 × 3 × 3 = 5 202</b> states × 17 actions = 88 434 Q-values per SU.</p></>],
  A: ['Action space', <><div className="eq"><span>a<sub>t</sub> ∈ {'{'}0, 1, …, K{'}'}</span></div><p style={{ marginTop: 12 }}><b>0</b> = idle / back off. <b>k</b> = transmit opportunistically on channel k.</p><div className="row" style={{ gap: 4, marginTop: 8 }}>{Array.from({ length: 17 }, (_, i) => <span key={i} className="mono" style={{ width: 34, height: 34, display: 'grid', placeItems: 'center', borderRadius: 8, background: i === 0 ? 'rgba(148,163,184,.2)' : i === 5 ? 'var(--gr)' : 'rgba(76,141,255,.18)', color: i === 5 ? '#111' : undefined, fontSize: 12 }}>{i}</span>)}</div></>],
  P: ['Transition probability', <><div className="eq"><span>P(s′ | s, a)</span></div><p style={{ marginTop: 12 }}>Driven by the Markov ON/OFF transitions of each PU and by channel fading. Q-learning is model-free — the agent never needs P explicitly, it learns from sampled transitions.</p><div className="eq" style={{ fontStyle: 'normal', fontFamily: 'var(--mono)', fontSize: 14 }}><span>P(OFF→ON) = s<sub>j</sub>·η &nbsp; P(ON→OFF) = s<sub>j</sub>·(1−η)</span></div></>],
  R: ['Reward', <><div className="eq"><span>R = w₁T − ρ<sub>coll</sub>I<sub>coll</sub> − ρ<sub>sw</sub>H<sub>switch</sub> + P<sub>idle</sub></span></div><p style={{ marginTop: 12 }}>Pays for throughput and charges for collisions and switching. <Link style={{ color: 'var(--ql)' }} to="/learn/reward">Try it interactively →</Link></p></>],
  γ: ['Discount factor', <><div className="eq"><span>0 ≤ γ ≤ 1 · here γ = 0.95</span></div><p style={{ marginTop: 12 }}>Weighs immediate against future reward. With γ = 0.95 the effective horizon is 1/(1−γ) ≈ <b>20 slots</b>, so the agent values staying on a channel that will remain free.</p></>],
}

export default function System() {
  const [m, setM] = useState('S')
  return <div className="stack">
    <PageHead kicker="Proposed system model & MDP" title={<>Agent ↔ environment, <em className="hl">formalised</em></>} sub="Click any block of the framework to see what it does and where it lives in the code." />
    <Arch />
    <Card kicker="Markov decision process" title={<>Spectrum access as a discrete-time MDP <span className="serif" style={{ color: 'var(--ql)' }}>(S, A, P, R, γ)</span></>} right={<Tabs tabs={Object.keys(MDP).map(k => ({ id: k, label: k }))} value={m} onChange={setM} />}>
      <h3 style={{ marginBottom: 10 }}>{MDP[m][0]}</h3>{MDP[m][1]}
    </Card>
    <Channel />
  </div>
}
