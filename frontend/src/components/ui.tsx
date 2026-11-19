import { ReactNode, useState } from 'react'
import { NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'

export function Card({ title, kicker, right, children, className = '', flat = false }: { title?: ReactNode; kicker?: string; right?: ReactNode; children?: ReactNode; className?: string; flat?: boolean }) {
  return <section className={`card ${flat ? 'flat' : ''} ${className}`}>
    {(title || kicker || right) && <div className="card-h"><div>{kicker && <div className="k">{kicker}</div>}{title && <h2>{title}</h2>}</div>{right}</div>}
    {children}
  </section>
}
export function PageHead({ kicker, title, sub, right }: { kicker: string; title: ReactNode; sub?: ReactNode; right?: ReactNode }) {
  return <motion.div className="page-head" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .3 }}>
    <div><div className="kick">{kicker}</div><h1>{title}</h1>{sub && <p>{sub}</p>}</div>{right}
  </motion.div>
}
export function Stat({ label, value, sub, color }: { label: string; value: ReactNode; sub?: ReactNode; color?: string }) {
  return <div className="card stat"><div className="v" style={{ color }}>{value}</div><div className="l">{label}</div>{sub && <div className="s">{sub}</div>}</div>
}
export function Tabs({ tabs, value, onChange }: { tabs: { id: any; label: ReactNode }[]; value: any; onChange: (v: any) => void }) {
  return <div className="tabs" role="tablist">{tabs.map(t => <button key={t.id} role="tab" aria-selected={value === t.id} className={value === t.id ? 'on' : ''} onClick={() => onChange(t.id)}>{t.label}</button>)}</div>
}
export function Slider({ label, value, min, max, step, onChange, fmt, disabled }: { label: ReactNode; value: number; min: number; max: number; step: number; onChange: (v: number) => void; fmt?: (v: number) => string; disabled?: boolean }) {
  const p = ((value - min) / (max - min)) * 100
  return <div className="field"><label><span>{label}</span><output>{fmt ? fmt(value) : value}</output></label>
    <input type="range" min={min} max={max} step={step} value={value} disabled={disabled} style={{ ['--p' as any]: `${p}%` }} onChange={e => onChange(+e.target.value)} aria-label={typeof label === 'string' ? label : undefined} /></div>
}
export function NumField({ label, value, onChange, min, max, step, hint, disabled }: { label: ReactNode; value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; hint?: string; disabled?: boolean }) {
  return <div className="field"><label><span>{label}</span>{hint && <span className="faint">{hint}</span>}</label>
    <input type="number" value={Number.isFinite(value) ? value : ''} min={min} max={max} step={step ?? 'any'} disabled={disabled} onChange={e => onChange(e.target.value === '' ? NaN : +e.target.value)} /></div>
}
export function Toggle({ on, onChange, label }: { on: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return <span className={`toggle ${on ? 'on' : ''}`} role="switch" aria-checked={on} tabIndex={0} onClick={() => onChange(!on)} onKeyDown={e => (e.key === ' ' || e.key === 'Enter') && (e.preventDefault(), onChange(!on))}><i />{label}</span>
}
export function Seg({ opts, value, onChange }: { opts: { id: any; label: ReactNode }[]; value: any; onChange: (v: any) => void }) { return <Tabs tabs={opts} value={value} onChange={onChange} /> }
export function StatusPill({ s }: { s: string }) {
  const k = s === 'done' ? 'ok' : s === 'failed' || s === 'interrupted' ? 'bad' : s === 'running' || s === 'queued' ? 'info' : 'warn'
  return <span className={`pill ${k}`}>{s}</span>
}
export function Legend({ items }: { items: { name: string; color: string; dashed?: boolean }[] }) {
  return <div className="legend">{items.map(i => <span key={i.name}><i className={i.dashed ? 'd' : ''} style={i.dashed ? { borderColor: i.color } : { background: i.color }} />{i.name}</span>)}</div>
}
export function Empty({ title, children }: { title: string; children?: ReactNode }) { return <div className="empty"><h3>{title}</h3><div>{children}</div></div> }
export function Skeleton({ h = 180 }: { h?: number }) { return <div className="skeleton" style={{ height: h }} /> }
export function Callout({ kind = 'warn', children }: { kind?: 'warn' | 'info' | 'bad'; children: ReactNode }) { return <div className={`callout ${kind === 'warn' ? '' : kind}`}>{children}</div> }
export function Flip({ front, back, label }: { front: ReactNode; back: ReactNode; label: string }) {
  const [on, setOn] = useState(false)
  return <div className={`flipcard ${on ? 'on' : ''}`} role="button" tabIndex={0} aria-pressed={on} onClick={() => setOn(!on)} onKeyDown={e => (e.key === 'Enter' || e.key === ' ') && setOn(!on)}>
    <div className="in"><div className="f"><div className="mono" style={{ color: '#fca5a5', fontSize: 11, letterSpacing: '.12em', marginBottom: 8 }}>{label}</div>{front}<div className="note" style={{ position: 'absolute', bottom: 12 }}>Click to see our fix ↻</div></div>
      <div className="b"><div className="mono" style={{ color: '#fcd34d', fontSize: 11, letterSpacing: '.12em', marginBottom: 8 }}>OUR FIX</div>{back}</div></div></div>
}
export function Side({ to, icon, children }: { to: string; icon: ReactNode; children: ReactNode }) {
  return <NavLink to={to} end={to === '/'} className={({ isActive }) => `nav ${isActive ? 'active' : ''}`}>{icon}<span>{children}</span></NavLink>
}
export function RunPicker({ runs, value, onChange, onlyReady = true }: { runs: any[]; value?: number; onChange: (n: number) => void; onlyReady?: boolean }) {
  const list = onlyReady ? runs.filter(r => r.policy_path) : runs
  if (!list.length) return <span className="pill warn">no trained run yet</span>
  return <select value={value ?? ''} onChange={e => onChange(+e.target.value)} aria-label="Select trained run" style={{ minWidth: 230 }}>
    {list.map(r => { const env = JSON.parse(r.env_json); return <option key={r.id} value={r.id}>#{r.id} · {r.name} · N={env.N} · {r.episodes_done} ep</option> })}
  </select>
}
