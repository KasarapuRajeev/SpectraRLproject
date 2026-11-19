export type Json = any
const base = ''
export class ApiError extends Error { status: number; constructor(m: string, s: number) { super(m); this.status = s } }

function headers(extra: Record<string, string> = {}) {
  const h: Record<string, string> = { 'Content-Type': 'application/json', ...extra }
  try { const k = localStorage.getItem('spectrarl_api_key'); if (k) h['X-API-Key'] = k } catch { /* ignore */ }
  return h
}

export async function api<T = Json>(path: string, opts: { method?: string; body?: unknown } = {}): Promise<T> {
  const res = await fetch(base + path, { method: opts.method || (opts.body ? 'POST' : 'GET'), headers: headers(), body: opts.body ? JSON.stringify(opts.body) : undefined })
  if (res.status === 204) return undefined as T
  const text = await res.text()
  let data: any = undefined
  try { data = text ? JSON.parse(text) : undefined } catch { /* not json */ }
  if (!res.ok) {
    const d = data?.detail
    const msg = typeof d === 'string' ? d : Array.isArray(d) ? d.map((x: any) => `${(x.loc || []).slice(1).join('.')}: ${x.msg}`).join('; ') : res.statusText
    throw new ApiError(msg || `HTTP ${res.status}`, res.status)
  }
  return data as T
}

export const COLORS = { ql: '#4C8DFF', greedy: '#F5A524', random: '#B26BFF', pu: '#38BDF8', ok: '#34D399', bad: '#F87171' }
export const NAMES: Record<string, string> = { ql: 'Q-learning (MO-QL-DSA)', greedy: 'Greedy', random: 'Random' }

export const fmt = (v: number | null | undefined, d = 1) => (v == null || Number.isNaN(v) ? '–' : v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }))
export const pct = (v: number | null | undefined, d = 1) => (v == null ? '–' : `${(v * 100).toFixed(d)}%`)
export const smooth = (a: number[], w = 10) => a.map((_, i) => { const s = Math.max(0, i - w + 1); let t = 0; for (let j = s; j <= i; j++) t += a[j]; return t / (i - s + 1) })
