import { useEffect, useMemo, useRef, useState } from 'react'
import { Card, PageHead, RunPicker, Empty } from '../components/ui'
import { LineChartX } from '../components/charts'
import { api, fmt } from '../lib/api'
import { useApp, useAsync } from '../lib/hooks'

const L: [number, string, string][] = [
  [0, '<b>Init</b> Q(s,a) ← 0 ∀ s∈S, a∈A', 'Start with an empty Q-table: the agent knows nothing about any channel. (One table per SU, 5 202 × 17 entries.)'],
  [0, '<b>for</b> episode = 1 … E<sub>max</sub> <b>do</b>', 'Train for up to E_max episodes (1000 by default), each T_slots = 500 time slots long.'],
  [1, 'reset env; generate PU ON/OFF vector O₀', 'Every episode begins with a fresh random PU occupancy drawn from the Markov stationary distribution.'],
  [1, '<b>for</b> t = 1 … T<sub>slots</sub> <b>do</b>', 'Step through the slots of the episode.'],
  [2, 'observe S<sub>t</sub> = (O<sub>t</sub>, Q<sub>t</sub>, H<sub>t</sub>, C<sub>t−1</sub>)', 'Read the (compressed) state: sensed PU occupancy, queue, channel quality and last channel.'],
  [2, 'a<sub>t</sub> ← random <b>if</b> p &lt; ε <b>else</b> argmax<sub>a</sub> Q(S<sub>t</sub>,a)', 'ε-greedy: explore a uniformly random action with probability ε, otherwise exploit the best-known one (ties broken at random).'],
  [2, 'transmit (a<sub>t</sub>=1..K) or backoff (a<sub>t</sub>=0)', 'Act: transmit on the chosen channel or back off for this slot.'],
  [2, 'measure C<sub>i,a</sub>(t), detect PU collision', 'The simulator resolves the slot: Rayleigh-faded SINR → Shannon capacity, PU collision and SU–SU collision checks.'],
  [2, 'compute R<sub>t</sub> via Eq. (3); observe S<sub>t+1</sub>', 'Score the slot with the multi-objective reward and observe the next state.'],
  [2, '<b>Bellman update of Q(S<sub>t</sub>, a<sub>t</sub>)</b>', 'Nudge Q(s,a) toward R + γ·max Q(s′,·) by step size α. This is where learning happens — see the live example below.'],
  [2, 'C<sub>t−1</sub> ← a<sub>t</sub>;  S<sub>t</sub> ← S<sub>t+1</sub>', 'Remember the channel used, so the next reward can detect a switch.'],
  [1, '<b>end for</b>;  decay ε', 'After each episode, ε ← max(ε_min, ε·δ): the agent explores less over time.'],
  [0, '<b>end for</b> → <b>output</b> Q(s,a), policy π(s)', 'The learned policy: in each state, pick the action with the highest Q-value.'],
]

export default function Algorithm() {
  const { runs, runId, setRunId } = useApp()
  const [i, setI] = useState(0); const [play, setPlay] = useState(false); const timer = useRef<any>()
  useEffect(() => { if (!play) return; timer.current = setInterval(() => setI(x => { if (x >= L.length - 1) { setPlay(false); return x } return x + 1 }), 1800); return () => clearInterval(timer.current) }, [play])
  const [a, setA] = useState(0.1), [g, setG] = useState(0.95), [e0, setE0] = useState(1), [em, setEm] = useState(0.01), [dl, setDl] = useState(0.995)
  const sched = useMemo(() => { const out: any[] = []; let e = e0; for (let k = 0; k <= 1000; k++) { if (k % 5 === 0) out.push({ ep: k, eps: e }); e = Math.max(em, e * dl) } return out }, [e0, em, dl])
  const bell = useAsync(() => api(`/api/runs/${runId}/bellman?steps=14&seed=5`), [runId], !!runId)
  const rows: any[] = bell.data?.rows || []
  return <div className="stack">
    <PageHead kicker="Q-learning & Bellman optimisation" title={<>MO-QL-DSA, <em className="hl">step by step</em></>} sub="Step through Algorithm 1, then watch real Bellman updates from a learning episode of a trained agent." />
    <div className="grid g2">
      <Card kicker="Algorithm 1" title="Q-learning dynamic spectrum allocation">
        <div className="algo">{L.map(([ind, t], j) => <div key={j} className={j === i ? 'cur' : j < i ? 'done' : ''} style={{ paddingLeft: 12 + ind * 26 }} onClick={() => setI(j)} dangerouslySetInnerHTML={{ __html: t }} />)}</div>
        <div className="row" style={{ marginTop: 14 }}><button className="btn sm" disabled={i === 0} onClick={() => setI(i - 1)}>← Back</button><button className="btn sm primary" disabled={i === L.length - 1} onClick={() => setI(i + 1)}>Next step →</button>
          <button className="btn sm" onClick={() => { if (i === L.length - 1) setI(0); setPlay(!play) }}>{play ? '❚❚ Pause' : '▶ Auto-play'}</button></div>
      </Card>
      <div className="stack">
        <Card kicker={`Step ${i + 1} of ${L.length}`} title="What happens here"><p dangerouslySetInnerHTML={{ __html: L[i][2] }} /></Card>
        <Card kicker="Update rules" title="Bellman & ε-greedy decay">
          <div className="eq" style={{ fontSize: 19 }}><span>Q(s,a) ← Q(s,a) + α[R + γ max<sub>a′</sub> Q(s′,a′) − Q(s,a)]</span></div>
          <div className="eq" style={{ fontSize: 19, marginTop: 10 }}><span>ε ← max(ε<sub>min</sub>, ε · δ<sub>decay</sub>)</span></div>
          <div className="row" style={{ marginTop: 10 }}>{[['α', a, setA, 0.1], ['γ', g, setG, 0.95], ['ε₀', e0, setE0, 1], ['ε_min', em, setEm, 0.01], ['δ', dl, setDl, 0.995]].map(([n, v, f, d]: any) => <label key={n} className="field" style={{ margin: 0, width: 82 }}><span className="lab">{n}</span><input type="number" step="any" value={v} onChange={e => f(+e.target.value)} /></label>)}</div>
          <p className="note" style={{ marginTop: 8 }}>Paper defaults: α = 0.1, γ = 0.95, ε₀ = 1.0, ε_min = 0.01, δ = 0.995. Horizon 1/(1−γ) ≈ {fmt(1 / Math.max(1e-6, 1 - g), 0)} slots.</p>
        </Card>
      </div>
    </div>
    <div className="grid g2">
      <Card kicker="ε-greedy schedule" title="Exploration decays per episode" right={<span className="pill info">ε &lt; 0.05 after episode {sched.find(s => s.eps < 0.05)?.ep ?? '—'}</span>}>
        <LineChartX data={sched} xKey="ep" series={[{ key: 'eps', name: 'ε', color: '#F5A524' }]} xLabel="episode" yFmt={v => v.toFixed(2)} height={230} />
      </Card>
      <Card kicker="Live example" title="Real Bellman updates from SU 0" right={<RunPicker runs={runs} value={runId} onChange={setRunId} />}>
        {!runId ? <Empty title="No trained run yet">Train an agent first (Training studio).</Empty> : <div className="scroll-x" style={{ maxHeight: 320, overflowY: 'auto' }}>
          <table className="t"><thead><tr><th>t</th><th>state</th><th>a</th><th className="n">R</th><th className="n">Q old</th><th className="n">Q new</th><th className="n">ΔQ</th></tr></thead><tbody>
            {rows.map(r => <tr key={r.t}><td className="mono">{r.t}</td><td className="mono" title={`C_{t-1}=${r.c_prev}, PU sensed on it=${r.occupied}, best free=${r.best_free}, queue bucket=${r.queue_bucket}`}>{r.state}</td><td>{r.action === 0 ? 'idle' : `ch ${r.action}`}{r.explored ? ' ·ε' : ''}</td>
              <td className="n">{fmt(r.reward, 2)}</td><td className="n">{fmt(r.q_old, 3)}</td><td className="n">{fmt(r.q_new, 3)}</td><td className="n" style={{ color: r.q_new - r.q_old >= 0 ? '#7ee8bf' : '#fca5a5' }}>{r.q_new - r.q_old >= 0 ? '+' : ''}{fmt(r.q_new - r.q_old, 3)}</td></tr>)}</tbody></table></div>}
        <p className="note" style={{ marginTop: 8 }}>Each row's Q new = Q old + α(R + γ·max Q(s′,·) − Q old), applied one slot later when s′ is observed. The simulator's unit tests verify this identity exactly. "·ε" marks exploratory actions.</p>
      </Card>
    </div>
  </div>
}
