import { useState } from 'react'
import { Card, PageHead, Slider, Seg, Toggle, Callout } from '../components/ui'
import { api, fmt } from '../lib/api'
import { useApp, useAsync } from '../lib/hooks'

const SCEN: [string, any][] = [
  ['Clean transmission', { action: 'transmit', capacity_ratio: 0.7 }],
  ['Transmission after a switch', { action: 'transmit', capacity_ratio: 0.7, switched: true }],
  ['Hit an active PU', { action: 'transmit', pu_active: true }],
  ['Collide with another SU', { action: 'transmit', su_collision: true }],
  ['Back off, all channels busy', { action: 'backoff', all_busy: true }],
  ['Back off, white space free', { action: 'backoff', all_busy: false }],
]

export default function RewardLab() {
  const { meta } = useApp()
  const d = meta?.defaults?.ql
  const [action, setAction] = useState<'transmit' | 'backoff'>('transmit')
  const [pu, setPu] = useState(false), [su, setSu] = useState(false), [sw, setSw] = useState(false), [busy, setBusy] = useState(false)
  const [cap, setCap] = useState(0.7)
  const [w, setW] = useState({ rho_collision: 1.5, rho_switch: 0.3, rho_su_collision: 0.5, idle_reward: 0.1, idle_penalty: 0.2, w_throughput: 1 })
  const ql = { ...(d || {}), ...w }
  const key = JSON.stringify([action, pu, su, sw, busy, cap, w])
  const one = useAsync(() => api('/api/reward', { body: { action, pu_active: pu, su_collision: su, switched: sw, all_busy: busy, capacity_ratio: cap, ql } }), [key], !!d)
  const mat = useAsync(() => Promise.all(SCEN.map(([, b]) => api('/api/reward', { body: { ...b, ql } }))), [JSON.stringify(w)], !!d)
  const r = one.data
  const terms = r ? [['T', 'throughput', r.throughput, '#4C8DFF'], ['I', 'collision', -r.collision, '#F87171'], ['H', 'switching', -r.switch, '#F5A524'], ['P', 'idle', r.idle, '#34D399']] as const : []
  const set = (k: string) => (v: number) => setW(s => ({ ...s, [k]: v }))
  return <div className="stack">
    <PageHead kicker="Capacity & multi-objective reward" title={<>Try the reward: <em className="hl">one decision, four terms</em></>}
      sub="Every number on this page is computed by the same compiled function the simulator uses to train agents (POST /api/reward) — not a re-implementation in the browser." />
    <div className="eq" style={{ background: 'rgba(76,141,255,.08)', fontSize: 24 }}><span>R = w₁<i style={{ color: '#93C5FD' }}>T</i> − ρ<sub>coll</sub><i style={{ color: '#FCA5A5' }}>I<sub>coll</sub></i> − ρ<sub>sw</sub><i style={{ color: '#FCD34D' }}>H<sub>switch</sub></i> + <i style={{ color: '#86EFAC' }}>P<sub>idle</sub></i></span><span className="tg">Eq. (3)</span></div>
    <div className="grid g2">
      <Card kicker="Set up a time slot" title="Scenario">
        <div style={{ marginBottom: 14 }}><Seg opts={[{ id: 'transmit', label: 'Transmit' }, { id: 'backoff', label: 'Back off' }]} value={action} onChange={setAction} /></div>
        {action === 'transmit' ? <div className="stack" style={{ gap: 12 }}>
          <Toggle on={pu} onChange={setPu} label="A PU is active on the chosen channel" /><Toggle on={su} onChange={setSu} label="Another SU transmitted on the same channel" /><Toggle on={sw} onChange={setSw} label="Switched from the previous channel" />
          <Slider label={<>Normalised capacity C / C<sub>max</sub></>} value={cap} min={0} max={1} step={0.05} onChange={setCap} fmt={v => v.toFixed(2)} /></div>
          : <Toggle on={busy} onChange={setBusy} label="Every channel is occupied by a PU" />}
        <div className="divider" />
        <div className="lab note" style={{ marginBottom: 8 }}>Reward weights (configurable; the deck does not publish its values)</div>
        <div className="grid g2" style={{ columnGap: 20 }}>
          <Slider label="w₁ throughput" value={w.w_throughput} min={0} max={3} step={0.1} onChange={set('w_throughput')} fmt={v => v.toFixed(1)} />
          <Slider label="ρ_coll (PU collision)" value={w.rho_collision} min={0} max={8} step={0.25} onChange={set('rho_collision')} fmt={v => v.toFixed(2)} />
          <Slider label="ρ_switch" value={w.rho_switch} min={0} max={2} step={0.05} onChange={set('rho_switch')} fmt={v => v.toFixed(2)} />
          <Slider label="ρ_su (SU collision, extension)" value={w.rho_su_collision} min={0} max={3} step={0.1} onChange={set('rho_su_collision')} fmt={v => v.toFixed(2)} />
          <Slider label="idle bonus (all busy)" value={w.idle_reward} min={0} max={1} step={0.05} onChange={set('idle_reward')} fmt={v => v.toFixed(2)} />
          <Slider label="idle penalty (white space)" value={w.idle_penalty} min={0} max={1} step={0.05} onChange={set('idle_penalty')} fmt={v => v.toFixed(2)} />
        </div>
      </Card>
      <Card kicker="Reward for this slot" title="Decomposition">
        <div className="stack" style={{ gap: 14 }}>
          {terms.map(([n, l, v, c]) => <div key={n} style={{ display: 'grid', gridTemplateColumns: '34px 1fr 74px', gap: 12, alignItems: 'center' }}>
            <span className="serif" style={{ fontSize: 24, fontStyle: 'italic', color: c }}>{n}</span>
            <div style={{ height: 22, background: 'rgba(148,163,184,.12)', borderRadius: 6, position: 'relative' }} title={l}>
              <div style={{ position: 'absolute', top: 0, bottom: 0, left: '50%', width: 2, background: 'rgba(148,163,184,.5)' }} />
              <div style={{ position: 'absolute', top: 0, bottom: 0, background: c, borderRadius: 5, transition: 'all .25s', ...(v >= 0 ? { left: '50%', width: `${Math.min(50, Math.abs(v) / 3 * 50)}%` } : { right: '50%', width: `${Math.min(50, Math.abs(v) / 3 * 50)}%` }) }} /></div>
            <span className="mono" style={{ textAlign: 'right' }}>{v >= 0 ? '+' : '−'}{Math.abs(v).toFixed(2)}</span></div>)}
        </div>
        <div className="divider" />
        <div className="row"><span className="muted">Total R<sub>t</sub></span><span className="sp" /><span className="serif" style={{ fontSize: 44, color: (r?.R ?? 0) >= 0 ? '#7ee8bf' : '#fca5a5' }}>{r ? `${r.R >= 0 ? '+' : '−'}${Math.abs(r.R).toFixed(2)}` : '…'}</span></div>
        <Callout kind="info"><b>Strict-paper mode:</b> set ρ_su = 0 and an SU–SU collision only zeroes the throughput term (T = 0), exactly as in the deck. Our experiments show why that is problematic — see the ablation on Benchmarks.</Callout>
      </Card>
    </div>
    <Card kicker="Compare" title="Reward of canonical situations under the current weights">
      <div className="scroll-x"><table className="t"><thead><tr><th>Situation</th><th className="n">Reward</th><th>Incentive</th></tr></thead><tbody>
        {SCEN.map(([n], i) => { const v = mat.data?.[i]?.R; return <tr key={n}><td>{n}</td><td className="n" style={{ color: v == null ? undefined : v >= 0 ? '#7ee8bf' : '#fca5a5' }}>{v == null ? '…' : `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)}`}</td>
          <td style={{ width: '45%' }}><div className="bar"><div style={{ width: `${v == null ? 0 : Math.max(2, Math.min(100, (v + 2) / 3 * 100))}%`, background: v != null && v < 0 ? 'linear-gradient(90deg,#ef4444,#f59e0b)' : undefined }} /></div></td></tr> })}
      </tbody></table></div>
      <p className="note" style={{ marginTop: 10 }}>A well-shaped reward ranks these situations sensibly: clean transmission ≫ smart back-off &gt; collisions. Note that with ρ_su = 0 a silent SU–SU collision (0.00) outranks idling (−0.20) — the agent has no reason to leave.</p>
    </Card>
  </div>
}
