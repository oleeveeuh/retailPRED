/** Vintage revision explorer — the same calendar month, value by vintage.
 *
 * Scrub the as-of position to see exactly what a real-time forecaster could
 * have known, versus what the latest revision eventually said.
 */

import { FC, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { SERIES_LABELS } from '../../content/lab';
import { vintageRevisions } from '../../content/labData';

const INK = '#25251f';
const INK2 = '#52514e';
const MUTED = '#77776c';
const GRID = '#d8d1c4';
const CRIMSON = '#a43b36';
const SAGE = '#89927a';

const fmt = (value: number) => value.toLocaleString('en-US', { maximumFractionDigits: 0 });

export const VintageExplorer: FC = () => {
  const [monthIndex, setMonthIndex] = useState(0);
  const [seriesId, setSeriesId] = useState(vintageRevisions.months[0].categories[0].series_id);
  const month = vintageRevisions.months[monthIndex];
  const category = month.categories.find((c) => c.series_id === seriesId) ?? month.categories[0];
  const [asOfIndex, setAsOfIndex] = useState(category.vintages.length - 1);
  const index = Math.min(asOfIndex, category.vintages.length - 1);

  const current = category.vintages[index];
  const revisionPct = (100 * (category.latest_revised - current.value)) / current.value;
  const data = category.vintages.map((v, i) => ({
    i,
    date: v.vintage_date,
    value: v.value,
  }));

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {vintageRevisions.months.map((m, i) => (
          <button
            key={m.month}
            onClick={() => {
              setMonthIndex(i);
              setAsOfIndex(Number.MAX_SAFE_INTEGER);
            }}
            className={`rounded-sm border px-3 py-1.5 font-mono text-xs lowercase transition-colors ${
              i === monthIndex
                ? 'border-[#89927a] bg-[#dfe1d5] text-[#25251f]'
                : 'border-stone-200 bg-white text-stone-600 hover:border-[#89927a]'
            }`}
          >
            {m.month.slice(0, 7)} · {m.label}
          </button>
        ))}
        <select
          value={seriesId}
          onChange={(e) => setSeriesId(e.target.value)}
          className="rounded-sm border border-stone-200 bg-white px-2 py-1.5 font-mono text-xs lowercase text-stone-600"
          aria-label="category"
        >
          {month.categories.map((c) => (
            <option key={c.series_id} value={c.series_id}>
              {SERIES_LABELS[c.series_id] ?? c.series_id}
            </option>
          ))}
        </select>
      </div>

      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 12, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke={GRID} strokeWidth={0.8} vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fontSize: 10, fill: MUTED }}
              stroke={GRID}
              tickFormatter={(d: string) => d.slice(2, 7)}
            />
            <YAxis
              domain={['dataMin - 300', 'dataMax + 300']}
              tick={{ fontSize: 10, fill: MUTED }}
              tickFormatter={fmt}
              stroke={GRID}
              width={62}
            />
            <Tooltip
              contentStyle={{ background: '#fbf9f4', border: `1px solid ${GRID}`, borderRadius: 2, fontSize: 12 }}
              formatter={(v: number | string) => (typeof v === 'number' ? fmt(v) : v)}
              labelFormatter={(d: string) => `vintage published ${d}`}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke={INK2}
              strokeWidth={2}
              dot={{ r: 2.5, fill: INK2, strokeWidth: 0 }}
              isAnimationActive={false}
              name="published value"
            />
            <ReferenceDot x={data[index]?.date} y={current.value} r={6} fill={CRIMSON} stroke="#fbf9f4" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 flex items-center gap-4">
        <input
          type="range"
          min={0}
          max={category.vintages.length - 1}
          value={index}
          onChange={(e) => setAsOfIndex(Number(e.target.value))}
          className="h-1 w-full cursor-pointer accent-[#a43b36]"
          aria-label="as-of vintage position"
        />
        <span className="w-24 shrink-0 font-mono text-xs lowercase text-stone-500">
          as-of {current.vintage_date}
        </span>
      </div>

      <div className="mt-3 grid gap-2 sm:grid-cols-3">
        <div className="rounded-sm border border-stone-200 bg-white px-3 py-2">
          <p className="font-mono text-[10px] lowercase text-stone-500">what you knew at as-of</p>
          <p className="font-mono text-sm" style={{ color: CRIMSON }}>
            {fmt(current.value)}
          </p>
        </div>
        <div className="rounded-sm border border-stone-200 bg-white px-3 py-2">
          <p className="font-mono text-[10px] lowercase text-stone-500">latest revision</p>
          <p className="font-mono text-sm" style={{ color: INK2 }}>
            {fmt(category.latest_revised)}
          </p>
        </div>
        <div className="rounded-sm border border-stone-200 bg-white px-3 py-2">
          <p className="font-mono text-[10px] lowercase text-stone-500">distance to final</p>
          <p className="font-mono text-sm" style={{ color: Math.abs(revisionPct) < 0.5 ? SAGE : CRIMSON }}>
            {revisionPct >= 0 ? '+' : ''}
            {revisionPct.toFixed(2)}%
          </p>
        </div>
      </div>

      <p className="mt-3 font-mono text-xs lowercase text-stone-500">
        {month.month.slice(0, 7)} ({month.label}) · showing first {category.vintages.length} of{' '}
        {category.n_vintages_total} published vintages · a month's first value appears ~45 days after
        month-end; every later vintage may revise it · {vintageRevisions.provenance.rule}
      </p>
    </div>
  );
};
