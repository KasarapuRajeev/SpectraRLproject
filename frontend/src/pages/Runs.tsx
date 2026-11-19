import { Link, useNavigate } from 'react-router-dom'
import { Download, Trash2 } from 'lucide-react'
import { Card, Empty, PageHead, StatusPill } from '../components/ui'
import { api, fmt, pct } from '../lib/api'
import { useApp, usePoll, useToast } from '../lib/hooks'

export default function Runs() {
  const { runs, refreshRuns, runId, setRunId } = useApp(); const toast = useToast(); const nav = useNavigate()
  const exps = usePoll(() => api<any[]>('/api/experiments'), 4000)
  const del = async (id: number) => { if (!confirm(`Delete run #${id}?`)) return; try { await api(`/api/runs/${id}`, { method: 'DELETE' }); refreshRuns(); toast.push('Run deleted', 'ok') } catch (e: any) { toast.push(e.message, 'err') } }
  const delE = async (id: number) => { if (!confirm(`Delete experiment #${id}?`)) return; try { await api(`/api/experiments/${id}`, { method: 'DELETE' }); exps.reload() } catch (e: any) { toast.push(e.message, 'err') } }
  return <div className="stack">
    <PageHead kicker="Persistence" title={<>Runs &amp; <em className="hl">models</em></>} sub="Every training run, its per-episode metrics and its Q-tables are stored in SQLite. Download a policy as a compressed .npz, or select it for the monitor and inspector." />
    <Card kicker="Training runs" title={`${runs.length} stored`}>
      {!runs.length ? <Empty title="Nothing trained yet">Start from the <Link to="/lab/train" style={{ color: 'var(--ql)' }}>Training studio</Link>.</Empty> : <div className="scroll-x"><table className="t"><thead><tr><th>#</th><th>Name</th><th>Status</th><th className="n">N</th><th className="n">η</th><th className="n">α</th><th className="n">Episodes</th><th className="n">QL reward</th><th className="n">QL thr.</th><th></th></tr></thead><tbody>
        {runs.map(r => { const e = JSON.parse(r.env_json), q = JSON.parse(r.ql_json), s = r.summary_json ? JSON.parse(r.summary_json).ql : null
          return <tr key={r.id} className={r.id === runId ? 'sel' : ''}><td className="mono">{r.id}</td><td>{r.name}{r.id === runId && <span className="pill info" style={{ marginLeft: 8 }}>selected</span>}</td><td><StatusPill s={r.status} /></td><td className="n">{e.N}</td><td className="n">{e.eta_pu}</td><td className="n">{q.alpha}</td><td className="n">{r.episodes_done}/{r.episodes_total}</td><td className="n">{s ? fmt(s.reward, 1) : '–'}</td><td className="n">{s ? `${fmt(s.throughput_mbps, 0)} Mbps` : '–'}</td>
            <td><div className="row" style={{ gap: 6, justifyContent: 'flex-end' }}>{r.policy_path && <><button className="btn sm" onClick={() => { setRunId(r.id); nav('/lab/monitor') }}>Monitor</button><button className="btn sm" onClick={() => { setRunId(r.id); nav('/lab/inspector') }}>Inspect</button><a className="btn sm" href={`/api/runs/${r.id}/policy`} aria-label="Download policy"><Download size={13} /></a></>}
              {r.status !== 'running' && r.status !== 'queued' && <button className="btn sm" onClick={() => del(r.id)} aria-label="Delete run"><Trash2 size={13} /></button>}</div></td></tr> })}</tbody></table></div>}
    </Card>
    <Card kicker="Experiments" title={`${(exps.data || []).length} stored`}>
      <div className="scroll-x"><table className="t"><thead><tr><th>#</th><th>Suite</th><th>Mode</th><th>Status</th><th className="n">Progress</th><th>Finished</th><th></th></tr></thead><tbody>
        {(exps.data || []).map(e => <tr key={e.id}><td className="mono">{e.id}</td><td>{e.suite}</td><td>{e.quick ? 'quick' : 'full scale'}</td><td><StatusPill s={e.status} /></td><td className="n">{fmt(e.progress * 100, 0)}%</td><td className="note">{e.finished_at ? new Date(e.finished_at * 1000).toLocaleString() : '–'}</td><td style={{ textAlign: 'right' }}>{e.status !== 'running' && <button className="btn sm" onClick={() => delE(e.id)} aria-label="Delete experiment"><Trash2 size={13} /></button>}</td></tr>)}</tbody></table></div>
    </Card>
  </div>
}
