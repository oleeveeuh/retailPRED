/** Window × category explorer — per-category WAPE by evaluation window. */

import { FC, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { MODEL_COLORS, MODEL_LABELS, SERIES_LABELS, WINDOW_LABELS } from '../../content/lab';
import { windowCategories } from '../../content/labData';

const INK = '#25251f';
const INK2 = '#52514e';
const MUTED = '#77776c';
const GRID = '#d8d1c4';

const pct = (value: number) => `${(100 * value).toFixed(2)}%`;

export const WindowCategoryExplorer: FC = () => {
  const [window, setWindow] = useState(windowCategories.windows[0]);
  const [active, setActive] = useState<string[]>(['seasonal_naive', 'theta_method']);

  const models = useMemo(
    () =>
      windowCategories.models.filter(
        (m) => active.includes(m) && !(m === 'regime_switch_policy' && window !== 'report_2021_2022'),
      ),
    [active, window],
  );

  const data = useMemo(() => {
    const bySeries = new Map<string, Record<string, number>>();
    for (const row of windowCategories.rows) {
      if (row.window !== window || !models.includes(row.model)) continue;
      bySeries.set(row.series_id, { ...(bySeries.get(row.series_id) ?? {}), [row.model]: 100 * row.wape });
    }
    return [...bySeries.entries()].map(([series_id, values]) => ({ series_id, ...values }));
  }, [window, models]);

  const toggleModel = (model: string) => {
    setActive((prev) => (prev.includes(model) ? prev.filter((m) => m !== model) : [...prev, model]));
  };

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-center gap-2">
        {windowCategories.windows.map((w) => (
          <button
            key={w}
            onClick={() => setWindow(w)}
            className={`rounded-sm border px-3 py-1.5 font-mono text-xs lowercase transition-colors ${
              w === window
                ? 'border-[#89927a] bg-[#dfe1d5] text-[#25251f]'
                : 'border-stone-200 bg-white text-stone-600 hover:border-[#89927a]'
            }`}
          >
            {WINDOW_LABELS[w]}
          </button>
        ))}
        <span className="mx-1 h-5 w-px bg-stone-200" />
        {windowCategories.models.map((model) => {
          const disabled = model === 'regime_switch_policy' && window !== 'report_2021_2022';
          const on = active.includes(model) && !disabled;
          return (
            <button
              key={model}
              onClick={() => !disabled && toggleModel(model)}
              disabled={disabled}
              title={
                model === 'regime_switch_policy'
                  ? 'the policy was evaluated on the report window only (single frozen pass)'
                  : undefined
              }
              className={`flex items-center gap-1.5 rounded-sm border px-3 py-1.5 font-mono text-xs lowercase transition-colors ${
                disabled
                  ? 'cursor-not-allowed border-stone-100 bg-stone-50 text-stone-300'
                  : on
                    ? 'border-stone-300 bg-white text-[#25251f]'
                    : 'border-stone-200 bg-white text-stone-400 hover:border-stone-300'
              }`}
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-sm"
                style={{ background: on ? MODEL_COLORS[model] : 'transparent', border: `1px solid ${MODEL_COLORS[model]}` }}
              />
              {MODEL_LABELS[model]}
              {disabled && ' (report only)'}
            </button>
          );
        })}
      </div>

      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 12, right: 16, bottom: 4, left: 0 }} barGap={1}>
            <CartesianGrid stroke={GRID} strokeWidth={0.8} vertical={false} />
            <XAxis
              dataKey="series_id"
              tickFormatter={(s: string) => SERIES_LABELS[s] ?? s}
              tick={{ fontSize: 10, fill: MUTED }}
              stroke={GRID}
              angle={-22}
              textAnchor="end"
              height={64}
              interval={0}
            />
            <YAxis
              tick={{ fontSize: 10, fill: MUTED }}
              tickFormatter={(v: number) => `${v.toFixed(0)}%`}
              stroke={GRID}
              width={52}
            />
            <Tooltip
              cursor={{ fill: 'rgba(137,146,122,0.08)' }}
              contentStyle={{ background: '#fbf9f4', border: `1px solid ${GRID}`, borderRadius: 2, fontSize: 12 }}
              formatter={(v: number | string) => (typeof v === 'number' ? pct(v / 100) : v)}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} formatter={(value: string) => MODEL_LABELS[value] ?? value} />
            {models.map((model) => (
              <Bar
                key={model}
                dataKey={model}
                fill={MODEL_COLORS[model]}
                radius={[2, 2, 0, 0]}
                isAnimationActive={false}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 flex flex-wrap gap-2">
        {models.map((model) => (
          <span key={model} className="rounded-sm border border-stone-200 bg-white px-3 py-1.5 font-mono text-xs lowercase">
            <span className="text-stone-500">mean-cat · {MODEL_LABELS[model]}:</span>{' '}
            <span style={{ color: INK2 }}>
              {pct(windowCategories.mean_category_wape[model]?.[window] ?? NaN)}
            </span>
          </span>
        ))}
      </div>

      <p className="mt-3 font-mono text-xs lowercase text-stone-500">
        subset measurements, window-labeled — never comparable to the official full-grid gate rows ·
        the policy exists on the report window only (single frozen evaluation) · the holdout is absent
        by design: it is never re-derived
      </p>
    </div>
  );
};
