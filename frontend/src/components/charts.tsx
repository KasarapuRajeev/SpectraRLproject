import { Area, CartesianGrid, ComposedChart, Legend as RLegend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis, Bar, BarChart, ReferenceLine } from 'recharts'
import { COLORS } from '../lib/api'

export type Series = { key: string; name: string; color: string; dashed?: boolean; band?: [string, string]; width?: number; dot?: boolean; hidden?: boolean }
const axis = { stroke: '#5B6788', fontSize: 11.5, tickLine: false as const, axisLine: { stroke: 'rgba(148,163,184,.25)' } }

function Tip({ active, payload, label, xFmt, yFmt }: any) {
  if (!active || !payload?.length) return null
  const rows = payload.filter((p: any) => p.value != null && !String(p.dataKey).startsWith('_') && !Array.isArray(p.value))
  return <div className="tooltip-box"><div className="mono faint" style={{ marginBottom: 4 }}>{xFmt ? xFmt(label) : label}</div>
    {rows.map((p: any) => <div className="r" key={p.dataKey}><span><span style={{ color: p.color }}>●</span> {p.name}</span><b>{yFmt ? yFmt(p.value) : p.value}</b></div>)}</div>
}

export function LineChartX({ data, xKey, series, height = 300, xLabel, yLabel, yFmt, xFmt, yDomain, xType = 'number', xTicks, refY }: {
  data: any[]; xKey: string; series: Series[]; height?: number; xLabel?: string; yLabel?: string; yFmt?: (v: number) => string; xFmt?: (v: any) => string; yDomain?: [any, any]; xType?: 'number' | 'category'; xTicks?: number[]; refY?: number }) {
  return <div style={{ width: '100%', height }}>
    <ResponsiveContainer>
      <ComposedChart data={data} margin={{ top: 8, right: 14, left: 0, bottom: xLabel ? 18 : 2 }}>
        <CartesianGrid stroke="rgba(148,163,184,.09)" vertical={false} />
        <XAxis dataKey={xKey} type={xType} domain={xType === 'number' ? ['dataMin', 'dataMax'] : undefined} ticks={xTicks} tickFormatter={xFmt} {...axis} label={xLabel ? { value: xLabel, position: 'insideBottom', offset: -10, fill: '#8B98B4', fontSize: 12 } : undefined} />
        <YAxis {...axis} domain={yDomain || ['auto', 'auto']} tickFormatter={yFmt} width={52} label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', fill: '#8B98B4', fontSize: 12, dx: 6 } : undefined} />
        <Tooltip content={<Tip xFmt={xFmt} yFmt={yFmt} />} cursor={{ stroke: 'rgba(255,255,255,.25)', strokeDasharray: '4 4' }} />
        {refY != null && <ReferenceLine y={refY} stroke="rgba(148,163,184,.4)" />}
        {series.filter(s => !s.hidden).map(s => <g key={s.key}>
          {s.band && <Area dataKey={`_band_${s.key}`} stroke="none" fill={s.color} fillOpacity={.14} isAnimationActive={false} legendType="none" name="" activeDot={false} />}
          <Line dataKey={s.key} name={s.name} stroke={s.color} strokeWidth={s.width ?? 2.4} strokeDasharray={s.dashed ? '6 5' : undefined} dot={s.dot ? { r: 3.5, fill: s.color, strokeWidth: 0 } : false} activeDot={{ r: 5 }} isAnimationActive={false} connectNulls />
        </g>)}
      </ComposedChart>
    </ResponsiveContainer></div>
}

/** attach error-band arrays `_band_<key>` = [lo, hi] */
export function withBands(rows: any[], series: Series[]) {
  return rows.map(r => { const o = { ...r }; series.forEach(s => { if (s.band && r[s.band[0]] != null) o[`_band_${s.key}`] = [r[s.key] - r[s.band[0]], r[s.key] + r[s.band[0]]] }); return o })
}

export function GroupedBars({ data, xKey, series, height = 300, yFmt, yLabel }: { data: any[]; xKey: string; series: Series[]; height?: number; yFmt?: (v: number) => string; yLabel?: string }) {
  return <div style={{ width: '100%', height }}><ResponsiveContainer>
    <BarChart data={data} margin={{ top: 8, right: 14, left: 0, bottom: 2 }} barGap={2}>
      <CartesianGrid stroke="rgba(148,163,184,.09)" vertical={false} />
      <XAxis dataKey={xKey} {...axis} />
      <YAxis {...axis} tickFormatter={yFmt} width={52} label={yLabel ? { value: yLabel, angle: -90, position: 'insideLeft', fill: '#8B98B4', fontSize: 12, dx: 6 } : undefined} />
      <Tooltip content={<Tip yFmt={yFmt} />} cursor={{ fill: 'rgba(255,255,255,.04)' }} />
      {series.map(s => <Bar key={s.key} dataKey={s.key} name={s.name} fill={s.color} radius={[4, 4, 0, 0]} isAnimationActive={false} />)}
    </BarChart></ResponsiveContainer></div>
}

export const POL_SERIES = (suffix = '', paper = false): Series[] => [
  { key: 'ql' + suffix, name: 'Q-learning', color: COLORS.ql, band: undefined }, { key: 'greedy' + suffix, name: 'Greedy', color: COLORS.greedy }, { key: 'random' + suffix, name: 'Random', color: COLORS.random }]
