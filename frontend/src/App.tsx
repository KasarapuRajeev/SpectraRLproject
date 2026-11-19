import { useEffect, useState } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { BookOpen, Cpu, FlaskConical, Gauge, LayoutDashboard, Menu, Radio, X } from 'lucide-react'
import { Side } from './components/ui'
import Section from './components/Section'
import { useApp } from './lib/hooks'
import Overview from './pages/Overview'
import Research from './pages/Research'
import System from './pages/System'
import RewardLab from './pages/RewardLab'
import Algorithm from './pages/Algorithm'
import Train from './pages/Train'
import Monitor from './pages/Monitor'
import Inspector from './pages/Inspector'
import Benchmarks from './pages/Benchmarks'
import Runs from './pages/Runs'
import Setup from './pages/Setup'

export default function App() {
  const [open, setOpen] = useState(false)
  const loc = useLocation()
  const { online, runs } = useApp()
  useEffect(() => { setOpen(false); window.scrollTo(0, 0) }, [loc.pathname])
  const running = runs.filter(r => r.status === 'running').length
  return <div className="app">
    <button className="menu-btn" aria-label="Toggle navigation" onClick={() => setOpen(!open)}>{open ? <X size={18} /> : <Menu size={18} />}</button>
    <aside className={`sidebar ${open ? 'open' : ''}`}>
      <div className="brand"><div className="logo"><Radio size={18} color="#fff" /></div><div><b>SpectraRL</b><small>MO-QL-DSA · cognitive radio</small></div></div>
      <Side to="/" icon={<LayoutDashboard size={17} />}>Home</Side>
      <Side to="/lab" icon={<FlaskConical size={17} />}>Lab<small className="sub">train · watch · inspect</small></Side>
      <Side to="/results" icon={<Gauge size={17} />}>Results<small className="sub">benchmarks · saved runs</small></Side>
      <Side to="/learn" icon={<BookOpen size={17} />}>Learn<small className="sub">how it works · scope</small></Side>
      <div className="sidefoot"><span className={`dot ${online ? '' : 'off'}`} />{online ? 'Backend online' : 'Backend unreachable'}<br /><Cpu size={12} style={{ verticalAlign: -2 }} /> {runs.length} runs{running ? ` · ${running} training` : ''}</div>
    </aside>
    <main className="main">
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/lab" element={<Section tabs={[{ to: '/lab/train', label: '1 · Train' }, { to: '/lab/monitor', label: '2 · Watch live' }, { to: '/lab/inspector', label: '3 · Inspect the policy' }]} />}>
          <Route index element={<Navigate to="train" replace />} /><Route path="train" element={<Train />} /><Route path="monitor" element={<Monitor />} /><Route path="inspector" element={<Inspector />} />
        </Route>
        <Route path="/results" element={<Section tabs={[{ to: '/results/benchmarks', label: 'Benchmarks' }, { to: '/results/runs', label: 'Saved runs' }]} />}>
          <Route index element={<Navigate to="benchmarks" replace />} /><Route path="benchmarks" element={<Benchmarks />} /><Route path="runs" element={<Runs />} />
        </Route>
        <Route path="/learn" element={<Section tabs={[{ to: '/learn/research', label: 'Background' }, { to: '/learn/system', label: 'System & MDP' }, { to: '/learn/reward', label: 'Reward' }, { to: '/learn/algorithm', label: 'Q-learning' }, { to: '/learn/setup', label: 'Setup & scope' }]} />}>
          <Route index element={<Navigate to="research" replace />} /><Route path="research" element={<Research />} /><Route path="system" element={<System />} />
          <Route path="reward" element={<RewardLab />} /><Route path="algorithm" element={<Algorithm />} /><Route path="setup" element={<Setup />} />
        </Route>
        {/* old URLs */}
        {[['train', '/lab/train'], ['monitor', '/lab/monitor'], ['inspector', '/lab/inspector'], ['benchmarks', '/results/benchmarks'], ['runs', '/results/runs'], ['research', '/learn/research'], ['system', '/learn/system'], ['reward', '/learn/reward'], ['algorithm', '/learn/algorithm'], ['setup', '/learn/setup']].map(([f, t]) => <Route key={f} path={'/' + f} element={<Navigate to={t} replace />} />)}
        <Route path="*" element={<div className="empty"><h3>Page not found</h3></div>} />
      </Routes>
    </main>
  </div>
}
