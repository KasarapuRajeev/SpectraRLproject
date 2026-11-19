import { useMemo, useState } from 'react'
import { Download, Zap } from 'lucide-react'
import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Callout, Card, Empty, PageHead, RunPicker, Seg, Stat, Toggle } from '../components/ui'
import { api, fmt, pct } from '../lib/api'
import { useApp, useAsync, useToast } from '../lib/hooks'

function Heat({ m, K, visited }: { m: number[][]; K: number; visited: number[] }) {
  const mx = Math.max(0.0001, ...m.flat().map(Math.abs))
  const cell = 26
  const col = (v: number) => v >= 0 ? `rgba(76,141,255,${Math.min(1, v / mx) * 0.95})` : `rgba(248,113,113,${Math.min(1, -v / mx) * 0.95})`
  return <div className="scroll-x"><svg width={44 + (K + 1) * cell} height={22 + (K + 1) * cell + 20} role="img" aria-label="Mean Q-value by previous channel and action">
    {Array.from({ length: K + 1 }, (_, a) => <text key={a} x={44 + a * cell + cell / 2} y={14} textAnchor="middle" fontSize={10} fill="#8B98B4" className="mono">{a}</text>)}
    {m.map((row, c) => <g key={c}>
      <text x={36} y={22 + c * cell + cell / 2 + 3} textAnchor="end" fontSize={10} fill={visited[c] ? '#8B98B4' : '#3b4663'} className="mono">{c === 0 ? 'idle' : c}</text>
      {row.map((v, a) => { const best = visited[c] > 0 && v === Math.max(...row)
        return <rect key={a} x={44 + a * cell + 1} y={22 + c * cell + 1} width={cell - 2} height={cell - 2} rx={4} fill={visited[c] ? col(v) : 'rgba(148,163,184,.05)'} stroke={best ? '#fff' : 'none'} strokeWidth={1.6}><title>{`C(t−1)=${c}, a=${a}: mean Q ${v.toFixed(3)} over ${visited[c]} visited states`}</title></rect> })}</g>)}
    <text x={44 + ((K + 1) * cell) / 2} y={22 + (K + 1) * cell + 14} textAnchor="middle" fontSize={11} fill="#8B98B4">action a (0 = back off, k = channel k) →</text>
  </svg></div>
}

export default function Inspector() {
  const { runs, runId, setRunId } = useApp()
  const toast = useToast()
  const [su, setSu] = useState(0), [cp, setCp] = useState(3), [occ, setOcc] = useState(false), [best, setBest] = useState(7), [qb, setQb] = useState('2'), [fb, setFb] = useState('2')
  const run = runs.find(r => r.id === runId)
  const env = run ? JSON.parse(run.env_json) : null
  const K = env?.K ?? 16, N = env?.N ?? 1
  const q = useAsync(() => api(`/api/runs/${runId}/qvalues?su=${su}&c_prev=${cp}&occupied=${occ ? 1 : 0}&best=${best}&queue_bucket=${qb}&free_bucket=${fb}`), [runId, su, cp, occ, best, qb, fb], !!runId)
  const pm = useAsync(() => api(`/api/runs/${runId}/policy-map?su=${su}`), [runId, su], !!runId)
  const [which, setWhich] = useState<'free' | 'occupied'>('free')
  const data = useMemo(() => (q.data?.q_values || []).map((v: number, a: number) => ({ a, v, best: a === q.data.action })), [q.data])
  const unvisited = q.data && q.data.q_values.every((v: number) => v === 0)
  const [edge, setEdge] = useState<any>(null)
  const edgeCall = async () => {
    const busy = Array.from({ length: K }, () => Math.random() < 0.4), snr = Array.from({ length: K }, () => -Math.log(1 - Math.random()) * 100)
    try { const d: any = await api(`/api/runs/${runId}/decide`, { body: { su, prev_channel: Math.floor(Math.random() * (K + 1)), sensed_busy: busy, snr, queue: 1 + Math.floor(Math.random() * 8) } }); setEdge({ ...d, busy, snr }) } catch (e: any) { toast.push(e.message, 'err') }
  }
  const b = pm.data?.behaviour
  return <div className="stack">
    <PageHead kicker="Interpretability & edge inference" title={<>Open the <em className="hl">Q-table</em></>} sub="Tabular Q-learning keeps its whole policy in a readable table. Pick a state and see exactly what the agent believes each action is worth."
      right={<div className="row"><RunPicker runs={runs} value={runId} onChange={setRunId} />{runId && <a className="btn sm" href={`/api/runs/${runId}/policy`}><Download size={13} />Export policy</a>}</div>} />
    {!runId ? <Card><Empty title="No trained policy yet">Train an agent first.</Empty></Card> : <>
      <div className="grid g-side" style={{ gridTemplateColumns: '340px minmax(0,1fr)', alignItems: 'start' }}>
        <Card kicker="State builder" title={<>S<sub>t</sub> = (C<sub>t−1</sub>, O, H, Q)</>}>
          <div className="field"><label><span>Secondary user</span></label><select value={su} onChange={e => setSu(+e.target.value)}>{Array.from({ length: N }, (_, i) => <option key={i} value={i}>SU {i}</option>)}</select></div>
          <div className="field"><label><span>Previous channel C<sub>t−1</sub></span></label><select value={cp} onChange={e => setCp(+e.target.value)}><option value={0}>0 — was idle</option>{Array.from({ length: K }, (_, i) => <option key={i} value={i + 1}>channel {i + 1}</option>)}</select></div>
          <div style={{ marginBottom: 14 }}><Toggle on={occ} onChange={setOcc} label="PU sensed on previous channel" /></div>
          <div className="field"><label><span>Best sensed-free channel (by SINR)</span></label><select value={best} onChange={e => setBest(+e.target.value)}><option value={0}>none free</option>{Array.from({ length: K }, (_, i) => <option key={i} value={i + 1}>channel {i + 1}</option>)}</select></div>
          <div className="field"><span className="lab">Queue backlog</span><Seg opts={[{ id: '0', label: 'empty' }, { id: '1', label: '1–3' }, { id: '2', label: '4+' }]} value={qb as any} onChange={setQb} /></div>
          <div className="field"><span className="lab">Sensed-free channels</span><Seg opts={[{ id: '0', label: '0' }, { id: '1', label: '1–4' }, { id: '2', label: '5+' }]} value={fb as any} onChange={setFb} /></div>
          {q.data && <p className="note mono">table row {q.data.state} of 5 202</p>}
        </Card>
        <Card kicker="Greedy policy π(s) = argmax Q(s,·)" title={q.data ? (q.data.action === 0 ? 'Back off this slot' : `Transmit on channel ${q.data.action}`) : '…'}
          right={q.data && <span className={`pill ${q.data.action === cp && cp > 0 ? 'ok' : q.data.action === 0 ? 'warn' : 'info'}`}>{q.data.action === cp && cp > 0 ? 'stay on current channel' : q.data.action === 0 ? 'idle' : 'switch'}</span>}>
          {unvisited && <Callout>This state was never visited during training (all Q-values are still 0), so the agent has no preference — it would pick uniformly at random. Try a state like C<sub>t−1</sub> = a channel, free, queue 4+.</Callout>}
          <div style={{ height: 300 }}><ResponsiveContainer><BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 14 }}>
            <XAxis dataKey="a" stroke="#5B6788" fontSize={11} tickLine={false} tickFormatter={v => (v === 0 ? 'idle' : v)} label={{ value: 'action', position: 'insideBottom', offset: -8, fill: '#8B98B4', fontSize: 11 }} /><YAxis stroke="#5B6788" fontSize={11} tickLine={false} width={46} tickFormatter={v => fmt(v, 1)} />
            <Tooltip cursor={{ fill: 'rgba(255,255,255,.05)' }} contentStyle={{ background: '#0b1224', border: '1px solid rgba(148,163,184,.3)', borderRadius: 10 }} formatter={(v: any) => [fmt(+v, 4), 'Q(s,a)']} labelFormatter={v => (v === 0 ? 'back off' : `channel ${v}`)} />
            <Bar dataKey="v" radius={[4, 4, 0, 0]} isAnimationActive={false}>{data.map((d: any, i: number) => <Cell key={i} fill={d.best ? '#F5A524' : d.v < 0 ? '#F87171' : '#4C8DFF'} fillOpacity={d.best ? 1 : 0.7} />)}</Bar></BarChart></ResponsiveContainer></div>
          <p className="note">Amber = chosen action. Positive Q means the agent expects good discounted return from that action in this state; negative means collisions, switching or idling cost more than throughput earns.</p>
        </Card>
      </div>
      <div className="grid g-main">
        <Card kicker="Policy map" title={<>Mean Q by previous channel × action (SU {su})</>} right={<Seg opts={[{ id: 'free', label: 'previous channel free' }, { id: 'occupied', label: 'PU on previous channel' }]} value={which} onChange={setWhich} />}>
          {pm.data ? <Heat m={pm.data[which].matrix} K={K} visited={pm.data[which].visited_states} /> : <div className="skeleton" style={{ height: 300 }} />}
          <p className="note" style={{ marginTop: 6 }}>Brighter blue = higher value; white outline = best action per row; averaged over visited states. When the previous channel is free the diagonal (stay) should dominate; when a PU returns the agent should leave it.</p>
        </Card>
        <div className="stack">
          <Card kicker="Measured behaviour" title="What the policy does (500-slot rollout)">
            {b ? <div className="grid g2" style={{ gap: 10 }}>
              <Stat label="stay on a channel that worked" value={pct(b.stay_after_success, 0)} color="#7ee8bf" /><Stat label="leave after a PU collision" value={pct(b.leave_after_pu_collision, 0)} />
              <Stat label="leave after an SU collision" value={pct(b.leave_after_su_collision, 0)} /><Stat label="idle slots" value={pct(b.idle_rate, 0)} /></div> : <div className="skeleton" style={{ height: 120 }} />}
          </Card>
          <Card kicker="Edge-ready" title="Single-decision inference" right={<button className="btn sm primary" onClick={edgeCall}><Zap size={13} />Random observation</button>}>
            {!edge ? <p className="note">Generates a random sensing/SINR observation and calls <code>POST /api/runs/{'{id}'}/decide</code>.</p> : <dl className="kv">
              <dt>Decision</dt><dd>{edge.action === 0 ? 'back off' : `transmit on ch ${edge.action}`}</dd><dt>Table row</dt><dd>{edge.state}</dd><dt>Server-side latency</dt><dd>{fmt(edge.latency_ms, 3)} ms</dd><dt>Model size / SU</dt><dd>{fmt(5202 * 17 * 4 / 1024, 0)} KB f32 · {fmt(5202 * 17 / 1024, 0)} KB int8</dd></dl>}
          </Card></div></div>
    </>}
  </div>
}
