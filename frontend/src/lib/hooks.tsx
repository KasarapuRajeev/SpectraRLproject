import { createContext, useCallback, useContext, useEffect, useRef, useState, ReactNode } from 'react'
import { api } from './api'

export function usePoll<T>(fn: () => Promise<T>, ms: number, deps: any[] = [], enabled = true) {
  const [data, setData] = useState<T | undefined>()
  const [error, setError] = useState<string | undefined>()
  const [loading, setLoading] = useState(true)
  const fnRef = useRef(fn); fnRef.current = fn
  const reload = useCallback(async () => {
    try { setData(await fnRef.current()); setError(undefined) } catch (e: any) { setError(e.message) } finally { setLoading(false) }
  }, [])
  useEffect(() => {
    if (!enabled) return
    let alive = true
    const tick = async () => { if (alive) await reload() }
    tick()
    const id = ms > 0 ? setInterval(tick, ms) : undefined
    return () => { alive = false; if (id) clearInterval(id) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ms, enabled, ...deps])
  return { data, error, loading, reload }
}

export function useAsync<T>(fn: () => Promise<T>, deps: any[] = [], enabled = true) {
  return usePoll(fn, 0, deps, enabled)
}

// ---- toasts ---------------------------------------------------------------
type Toast = { id: number; kind: 'ok' | 'err' | 'info'; text: string }
const ToastCtx = createContext<{ push: (t: string, k?: Toast['kind']) => void }>({ push: () => {} })
export const useToast = () => useContext(ToastCtx)
export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([])
  const push = useCallback((text: string, kind: Toast['kind'] = 'info') => {
    const id = Math.random(); setItems(s => [...s, { id, kind, text }]); setTimeout(() => setItems(s => s.filter(x => x.id !== id)), 5000)
  }, [])
  return <ToastCtx.Provider value={{ push }}>{children}
    <div className="toasts" role="status" aria-live="polite">{items.map(t => <div key={t.id} className={`toast ${t.kind}`}>{t.text}</div>)}</div>
  </ToastCtx.Provider>
}

// ---- selected run + meta --------------------------------------------------
type Meta = any
const AppCtx = createContext<{ meta?: Meta; runs: any[]; refreshRuns: () => void; runId?: number; setRunId: (n?: number) => void; online: boolean }>({ runs: [], refreshRuns: () => {}, setRunId: () => {}, online: true })
export const useApp = () => useContext(AppCtx)
export function AppProvider({ children }: { children: ReactNode }) {
  const meta = usePoll(() => api('/api/meta'), 0)
  const health = usePoll(() => api('/api/health'), 8000)
  const runsQ = usePoll(() => api<any[]>('/api/runs'), 4000)
  const [runId, setRunIdState] = useState<number | undefined>(() => { const v = localStorage.getItem('spectrarl_run'); return v ? +v : undefined })
  const setRunId = (n?: number) => { setRunIdState(n); if (n) localStorage.setItem('spectrarl_run', String(n)); else localStorage.removeItem('spectrarl_run') }
  const runs = runsQ.data || []
  useEffect(() => { // default to the newest finished run if the stored one is gone
    if (!runsQ.data) return
    const usable = runsQ.data.filter(r => r.policy_path)
    if (runId && !runsQ.data.find(r => r.id === runId) ) setRunId(usable[0]?.id)
    else if (!runId && usable.length) setRunId(usable[0].id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runsQ.data])
  return <AppCtx.Provider value={{ meta: meta.data, runs, refreshRuns: runsQ.reload, runId, setRunId, online: !health.error }}>{children}</AppCtx.Provider>
}
