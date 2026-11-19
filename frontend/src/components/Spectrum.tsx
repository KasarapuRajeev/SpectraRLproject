import { useEffect, useRef } from 'react'

export const CODE_COLORS: Record<number, string> = { 1: '#F5A524', 2: '#F87171', 3: '#B26BFF' }

/** Channel x time waterfall. occ[t][k] PU occupancy, act[t][i] SU action (0 idle, k+1 channel), code[t][i] 1 ok / 2 PU collision / 3 SU collision. */
export default function Spectrum({ occ, act, code, K, upto, height = 260, label, mini = false }: { occ: number[][]; act: number[][]; code: number[][]; K: number; upto: number; height?: number; label?: string; mini?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const cv = ref.current; if (!cv) return
    const draw = () => {
      const dpr = window.devicePixelRatio || 1
      const w = cv.clientWidth, h = cv.clientHeight
      if (cv.width !== w * dpr || cv.height !== h * dpr) { cv.width = w * dpr; cv.height = h * dpr }
      const ctx = cv.getContext('2d')!; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h)
      const T = occ.length; if (!T) return
      const padL = mini ? 0 : 34, padB = mini ? 0 : 18
      const cw = (w - padL) / T, ch = (h - padB) / K, gap = Math.min(1, cw * 0.12)
      ctx.font = '10px JetBrains Mono, monospace'; ctx.textBaseline = 'middle'
      for (let k = 0; k < K; k++) {
        if (!mini && (k % 2 === 0 || K <= 8)) { ctx.fillStyle = '#5B6788'; ctx.textAlign = 'right'; ctx.fillText(String(k + 1), padL - 7, k * ch + ch / 2) }
        for (let t = 0; t < T; t++) {
          const x = padL + t * cw, y = k * ch, played = t < upto
          ctx.fillStyle = played ? 'rgba(148,163,184,.07)' : 'rgba(148,163,184,.035)'
          ctx.fillRect(x + gap / 2, y + gap / 2, cw - gap, ch - gap)
          if (occ[t][k]) { ctx.fillStyle = played ? 'rgba(56,189,248,.42)' : 'rgba(56,189,248,.13)'; ctx.fillRect(x + gap / 2, y + gap / 2, cw - gap, ch - gap) }
        }
      }
      // SU transmissions
      for (let t = 0; t < Math.min(upto, T); t++) {
        const cnt: Record<number, { n: number; c: number }> = {}
        const sev: Record<number, number> = { 0: 0, 1: 1, 3: 2, 2: 3 }
        for (let i = 0; i < act[t].length; i++) { const a = act[t][i]; if (a > 0) { const e = (cnt[a] ||= { n: 0, c: 0 }); e.n++; if (sev[code[t][i]] > sev[e.c]) e.c = code[t][i] } }
        for (const a in cnt) {
          const k = +a - 1, e = cnt[a]; const x = padL + t * cw, y = k * ch
          ctx.fillStyle = CODE_COLORS[e.c] || '#F5A524'
          ctx.fillRect(x + gap / 2 + cw * 0.12, y + gap / 2 + ch * 0.14, Math.max(1, cw - gap - cw * 0.24), Math.max(2, ch - gap - ch * 0.28))
          if (e.n > 1 && cw > 7 && ch > 9) { ctx.fillStyle = '#10131f'; ctx.textAlign = 'center'; ctx.fillText(String(e.n), x + cw / 2, y + ch / 2 + 0.5) }
        }
      }
      if (upto > 0 && upto <= T) { ctx.strokeStyle = 'rgba(255,255,255,.85)'; ctx.lineWidth = 1.5; const x = padL + upto * cw; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h - padB); ctx.stroke() }
      if (!mini) { ctx.fillStyle = '#5B6788'; ctx.textAlign = 'left'; ctx.fillText('time slot →', padL, h - 7); ctx.save(); ctx.translate(9, (h - padB) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = 'center'; ctx.fillText('channel', 0, 0); ctx.restore() }
    }
    draw()
    const ro = new ResizeObserver(draw); ro.observe(cv)
    return () => ro.disconnect()
  }, [occ, act, code, K, upto, mini])
  return <canvas ref={ref} className="cv" style={{ height }} role="img" aria-label={label || 'Spectrum occupancy by channel and time slot'} />
}

export function SpectrumLegend() {
  return <div className="cell-legend">
    <span><i style={{ background: 'rgba(56,189,248,.5)' }} />Primary user active</span><span><i style={{ background: '#F5A524' }} />SU transmitting (success)</span>
    <span><i style={{ background: '#F87171' }} />PU collision</span><span><i style={{ background: '#B26BFF' }} />SU–SU collision</span><span><i style={{ background: 'rgba(148,163,184,.12)' }} />White space</span></div>
}
