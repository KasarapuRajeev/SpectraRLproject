import { useState } from 'react'
import { Link } from 'react-router-dom'
import { Card, Flip, PageHead, Callout } from '../components/ui'
import { useApp } from '../lib/hooks'

const CHALLENGES = [
  ['Random PU activity', 'Primary users switch ON and OFF unpredictably. The simulator models each PU as a Markov ON/OFF source, so an agent can learn which channels tend to stay free.', 'Markov chains in core.run_episode'],
  ['Multi-user contention', 'When every SU chases the same best channel they collide. Independent learners are penalised for SU–SU collisions and learn to spread across channels.', 'su_contention: domain | sinr'],
  ['Channel fading', 'Rayleigh fading makes link quality change every slot. SINR (Eq. 1) and Shannon capacity (Eq. 2) feed the throughput term of the reward.', 'h² ~ Exp(1) per SU × channel × slot'],
  ['Switching delays', 'A channel change costs RF re-tuning and protocol overhead. The reward carries an explicit switching penalty and the simulator removes a fraction of the slot on a switch.', 'rho_switch + switch_overhead'],
]
const GAPS = [
  ['Single-objective focus.', 'Most papers maximise SU throughput or minimise PU interference — rarely both.', 'One multi-objective reward that pays for throughput and charges for PU collisions in the same signal. See the ablation on the Benchmarks page.'],
  ['Heavy deep models.', 'Large compute, non-interpretable weights and long convergence times.', 'A tabular Q-learning agent on a compressed MDP state: readable Q-values (Policy inspector) and microsecond-scale decisions (Runtime benchmark).'],
  ['Ignored switching.', 'No handover cost → channel thrashing and poor QoS.', 'An explicit switching penalty keeps the agent on a good free channel instead of chasing small SINR gains. Measured handover reduction on Benchmarks.'],
]

export default function Research() {
  const { meta } = useApp()
  const lit: string[][] = meta?.paper?.literature || []
  const [ch, setCh] = useState(0)
  const [li, setLi] = useState<number | null>(null)
  const detail: Record<string, string> = {
    'Digital Twin + RL': 'Uses a digital twin to orchestrate spectrum in CRNs. High fidelity, but needs heavy computation.', 'Deep Q-Networks': 'A DQN allocates channels in vehicular CRNs. Handles mobility but needs lots of pre-training data.',
    'Federated RL': 'Keeps sensing data private, but non-IID sensing across users makes the shared policy drift.', 'Actor-Critic (DDPG)': 'Continuous power and spectrum control, but gradients become unstable under bursty PU traffic.',
    'Multi-Agent RL': 'Local reward sharing scales across IoT, but state/action spaces grow exponentially in dense networks.', PPO: 'Stable policy updates, but too much computation for edge radios.',
    'Hidden Markov Model': 'Predicts PU arrival times; false alarms trigger unnecessary handovers.', 'Simple Q-Learning': 'Easy to implement, but with no switching cost it thrashes between channels. (Implemented here as the throughput-only ablation.)',
    'Multi-objective QL': 'One tabular Q-learning agent whose reward balances throughput, PU interference and handovers; designed for discrete sub-channels.' }
  return <div className="stack">
    <PageHead kicker="Introduction, literature, gaps" title={<>Why dynamic spectrum allocation <em className="hl">needs to learn</em></>}
      sub="Rapid growth in wireless devices, IoT and ultra-dense 5G has created a severe spectrum shortage — yet licensed bands sit idle much of the time. Cognitive Radio lets Secondary Users (SUs) opportunistically use spectrum left unused by Primary Users (PUs). Random and greedy allocation cannot adapt to rising traffic." />
    <div className="grid g-main">
      <Card kicker="Why CRN allocation is hard" title="Four challenges that shape the design">
        <div className="grid g2">{CHALLENGES.map(([t, d, impl], i) => <button key={t} onClick={() => setCh(i)} className="card flat" style={{ textAlign: 'left', cursor: 'pointer', borderColor: ch === i ? 'rgba(76,141,255,.6)' : undefined, background: ch === i ? 'rgba(76,141,255,.1)' : undefined }}>
          <div className="mono" style={{ color: 'var(--ql)', fontSize: 12 }}>0{i + 1}</div><h3 style={{ margin: '4px 0' }}>{t}</h3></button>)}</div>
        <div className="divider" /><h3>{CHALLENGES[ch][0]}</h3><p style={{ marginTop: 6 }}>{CHALLENGES[ch][1]}</p><span className="pill info mono">{CHALLENGES[ch][2]}</span>
      </Card>
      <Card kicker="Our contribution" title="One lightweight answer">
        <p>A <b>lightweight MDP</b> solved by an <b>adaptive Q-Learning agent</b> with a <b>novel multi-objective reward</b> that jointly maximises throughput, protects PUs and suppresses spectral switching — fast enough for real-time edge devices.</p>
        <div className="row" style={{ marginTop: 10 }}><Link className="btn sm" to="/learn/system">System &amp; MDP</Link><Link className="btn sm" to="/learn/algorithm">Algorithm</Link></div>
      </Card>
    </div>
    <Card kicker="Table 1" title="Literature survey — what recent work offers and where it falls short" right={<span className="note">click a row</span>}>
      <div className="grid g-main"><div className="scroll-x"><table className="t"><thead><tr><th>Author (Year)</th><th>Journal</th><th>Methodology</th><th>Advantage</th><th>Disadvantage</th></tr></thead>
        <tbody>{lit.map((r, i) => <tr key={i} className={`click ${i === lit.length - 1 ? 'hl' : ''} ${li === i ? 'sel' : ''}`} onClick={() => setLi(i)}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody></table></div>
        <div className="card flat">{li == null ? <p className="muted">Select a row to see what each approach does and the trade-off that motivated this work.</p> : <>
          <div className="mono" style={{ color: 'var(--ql)', fontSize: 12 }}>{lit[li][2]}</div><h3 style={{ margin: '4px 0 8px' }}>{lit[li][0]}</h3><p>{detail[lit[li][2]]}</p>
          <div className="row"><span className="pill ok">+ {lit[li][3]}</span><span className="pill bad">− {lit[li][4]}</span></div></>}</div></div>
    </Card>
    <div><div className="page-head" style={{ marginBottom: 10 }}><div><div className="kick">Research gap</div><h2 style={{ fontSize: 21 }}>Three gaps in prior work</h2></div></div>
      <div className="grid g3">{GAPS.map(([t, d, fix], i) => <Flip key={t} label={`GAP ${i + 1}`} front={<><h3 style={{ marginBottom: 6 }}>{t}</h3><p>{d}</p></>} back={<p>{fix}</p>} />)}</div></div>
    <Callout kind="info"><b>Honesty note.</b> The deck's literature summary is reproduced as written. Where the presentation's own results slides state numbers, this project does not copy them — every result is recomputed by the simulator in this repository, and the deck's numbers are shown only as a labelled comparison on the Benchmarks page.</Callout>
  </div>
}
