import { Callout, Card, PageHead } from '../components/ui'
import { useApp } from '../lib/hooks'

const NOT_DONE: [string, string][] = [
  ['Multi-agent deep RL (MADRL) / federated learning', 'Listed as future work in the deck. Not implemented — the system uses independent tabular Q-learners, one per SU.'],
  ['Continuous power control', 'Action space is channel selection (+ stay-idle) only; transmit power is fixed at 23 dBm.'],
  ['Real radio hardware (USRP / GNU Radio)', 'No SDR available. The "edge" path is a per-slot /api/runs/{id}/decide call that takes a sensed observation and returns an action.'],
  ['Deck Fig. 7 image', 'The deck image for the α-sweep was not part of the supplied file; the α experiment is re-run and its curves are drawn from our own data.'],
  ['Absolute numbers from the deck', 'The authors\' simulator and several parameters are unpublished. Directions are comparable; magnitudes are not.'],
]

export default function Setup() {
  const { meta } = useApp()
  const e = meta?.defaults?.env, q = meta?.defaults?.ql
  const envRows: [string, any, string][] = e ? [
    ['Sub-channels K', e.K, 'Table 3'], ['Secondary users N', e.N, 'Table 3'], ['PUs M', e.M, 'Table 3'],
    ['Bandwidth / channel', `${e.bandwidth_hz / 1e6} MHz`, 'Table 3'], ['PU activity η', e.eta_pu, 'Table 3'],
    ['Tx power', `${e.ptx_dbm} dBm`, 'assumed'], ['Noise PSD', `${e.noise_dbm_hz} dBm/Hz`, 'thermal'], ['Path-loss exponent', e.path_loss_exp, 'assumed'],
    ['Carrier', `${e.fc_hz / 1e9} GHz`, 'assumed'], ['PU dynamics ×', e.pu_dynamics, 'assumed'], ['Arrival prob.', e.arrival_prob, 'assumed'],
    ['Sensing P_fa / P_md', `${e.p_false_alarm} / ${e.p_miss_detect}`, 'assumed'], ['Slot', `${e.slot_ms} ms`, 'assumed'], ['Switch overhead', e.switch_overhead, 'assumed'],
  ] : []
  const qlRows: [string, any, string][] = q ? [
    ['Learning rate α', q.alpha, 'Table 3'], ['Discount γ', q.gamma, 'Table 3'], ['ε start → min', `${q.eps0} → ${q.eps_min}`, 'Table 3'], ['ε decay / episode', q.eps_decay, 'Table 3'],
    ['Episodes', q.episodes, 'Table 3'], ['Slots / episode', q.slots, 'assumed'], ['ρ_coll (PU)', q.rho_collision, 'deck example'], ['ρ_switch', q.rho_switch, 'assumed'],
    ['ρ_su (SU–SU)', q.rho_su_collision, 'our extension'], ['Idle reward / penalty', `${q.idle_reward} / ${q.idle_penalty}`, 'assumed'],
  ] : []
  const tbl = (rows: [string, any, string][]) => <table className="t"><thead><tr><th>Parameter</th><th className="n">Value</th><th>Origin</th></tr></thead><tbody>{rows.map(r => <tr key={r[0]}><td>{r[0]}</td><td className="n mono">{r[1]}</td><td><span className={`pill ${r[2] === 'Table 3' ? 'ok' : ''}`}>{r[2]}</span></td></tr>)}</tbody></table>
  return <div className="stack">
    <PageHead kicker="Configuration" title={<>Parameters, <em className="hl">assumptions</em> &amp; limits</>} sub="What comes from the deck, what we had to choose ourselves, and what is deliberately not claimed." />
    <div className="grid g2"><Card kicker="Environment" title="Defaults">{tbl(envRows)}</Card><Card kicker="Learner" title="Defaults">{tbl(qlRows)}</Card></div>
    <Card kicker="Modelling choices" title="Assumptions the deck does not specify">
      <ol className="note" style={{ paddingLeft: 18, lineHeight: 1.7 }}>{(meta?.assumptions || []).map((a: string, i: number) => <li key={i}>{a}</li>)}</ol></Card>
    <Card kicker="Honest scope" title="Not implemented / not reproduced"><div className="stack">{NOT_DONE.map(([t, d]) => <div key={t}><b>{t}</b><div className="note">{d}</div></div>)}</div></Card>
    <Callout kind="info">Everything shown in this application is computed live by the backend (Numba simulation kernel + Q-learning trainer). No chart is a hard-coded or pre-recorded number; the deck's own values appear only where explicitly labelled as such.</Callout>
  </div>
}
