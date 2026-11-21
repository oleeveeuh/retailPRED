/** SHAP explorer — the shift diagnosis, then the challenger's own ledger.
 *
 * Top: the measured reversal (29.2% revised-history validation improvement
 * into the vintage gate's 4.1% degradation) and the attribution drift
 * between two fits of the identical architecture on revised vs point-in-time
 * values of the same months — the model-side account of the distribution
 * shift the documents describe.
 *
 * Bottom: today's refit challenger, explained in-sample with frozen
 * TreeExplainer values. Waterfalls close exactly to the model output; the
 * residual after the top-k bars is rendered as "all other features".
 */

import { FC, useMemo, useState } from 'react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { shapAnalysis } from '../../content/labData';

const INK = '#25251f';
const INK2 = '#52514e';
const MUTED = '#77776c';
const GRID = '#d8d1c4';
const POLICY = '#89927a';
const CRIMSON = '#a43b36';
const DOWN = '#1E342E';

const LABELS: Record<string, string> = {
  series_code: 'series identity',
  lag_1_ratio: 'lag 1 ÷ year-ago',
  lag_2_ratio: 'lag 2 ÷ year-ago',
  lag_3_ratio: 'lag 3 ÷ year-ago',
  lag_6_ratio: 'lag 6 ÷ year-ago',
  rolling_mean_3_ratio: '3-mo mean ratio',
  rolling_std_3_ratio: '3-mo std ratio',
  rolling_mean_6_ratio: '6-mo mean ratio',
  rolling_std_6_ratio: '6-mo std ratio',
  rolling_mean_12_ratio: '12-mo mean ratio',
  rolling_std_12_ratio: '12-mo std ratio',
  diff_1_ratio: '1-mo diff ratio',
  pct_change_1: '1-mo % change',
  month_sin: 'season (sin)',
  month_cos: 'season (cos)',
};

const label = (feature: string) => LABELS[feature] ?? feature;

/** SHAP values live in log-ratio units; the human-scale twin is % vs year-ago. */
const pctOf = (value: number) => `${value >= 0 ? '+' : ''}${((Math.exp(value) - 1) * 100).toFixed(1)}%`;

const logDelta = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;

const signedPct = (value: number) => `${value >= 0 ? '+' : ''}${(100 * value).toFixed(1)}%`;

const SubHeading: FC<{ children: string }> = ({ children }) => (
  <p className="mt-6 mb-3 border-t border-[#d8d1c4] pt-4 font-mono text-[11px] uppercase tracking-wide text-stone-500 first:mt-0 first:border-t-0 first:pt-0">
    {children}
  </p>
);

export const ShapImportanceExplorer: FC = () => {
  const { provenance, global_importance: globalImportance, waterfalls, shift_diagnosis: shift } =
    shapAnalysis;
  const [seriesId, setSeriesId] = useState(waterfalls[0]?.series_id ?? '');

  const waterfall = useMemo(
    () => waterfalls.find((item) => item.series_id === seriesId) ?? waterfalls[0],
    [waterfalls, seriesId],
  );

  const importanceData = useMemo(
    () =>
      globalImportance.map((row) => ({
        name: label(row.feature),
        share: 100 * row.share,
        shap: row.mean_abs_shap,
      })),
    [globalImportance],
  );

  /** Attribution drift, biggest move first (rows arrive sorted; recharts renders
   * vertical-layout categories bottom-up, so reverse for a top-down read). */
  const driftData = useMemo(
    () =>
      [...shift.rows]
        .reverse()
        .map((row) => ({
          name: label(row.feature),
          drift: row.drift_abs,
          revised: row.revised_history.mean_abs_shap,
          point: row.point_in_time.mean_abs_shap,
          valueShift: row.point_in_time.mean_value - row.revised_history.mean_value,
        })),
    [shift.rows],
  );

  /** Cumulative waterfall: top-k contributions plus the closing residual. */
  const waterfallData = useMemo(() => {
    if (!waterfall) return [];
    const rows: {
      name: string;
      offset: number;
      span: number;
      sign: number;
      shap: number;
    }[] = [];
    let cursor = waterfall.base_value;
    for (const contribution of waterfall.contributions) {
      const next = cursor + contribution.shap;
      rows.push({
        name: label(contribution.feature),
        offset: Math.min(cursor, next),
        span: Math.abs(contribution.shap),
        sign: Math.sign(contribution.shap),
        shap: contribution.shap,
      });
      cursor = next;
    }
    const residual = waterfall.model_output - cursor;
    if (Math.abs(residual) > 1e-9) {
      rows.push({
        name: 'all other features',
        offset: Math.min(cursor, cursor + residual),
        span: Math.abs(residual),
        sign: Math.sign(residual),
        shap: residual,
      });
    }
    return rows;
  }, [waterfall]);

  const waterfallDomain = useMemo(() => {
    if (!waterfall || waterfallData.length === 0) return [0, 1];
    // A mid-sequence contribution can overshoot the final output, so the
    // domain must cover every segment's start and end, not just the endpoints.
    const points = [waterfall.base_value, waterfall.model_output];
    for (const row of waterfallData) {
      points.push(row.offset, row.offset + row.span);
    }
    const low = Math.min(...points);
    const high = Math.max(...points);
    const pad = Math.max((high - low) * 0.08, 1e-6);
    return [low - pad, high + pad];
  }, [waterfall, waterfallData]);

  if (!waterfall) {
    return (
      <p className="font-mono text-xs lowercase text-stone-500">
        shap bundle is empty — regenerate it with `make shap-data`.
      </p>
    );
  }

  const validation = shift.provenance.references.development_validation;
  const gate = shift.provenance.references.vintage_gate;

  return (
    <div>
      <SubHeading>the reversal — measured, from the frozen documents' own artifacts</SubHeading>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Readout
          label="validation, revised history"
          value={signedPct(validation.relative_improvement)}
          title={`${validation.label}: naive ${(100 * validation.naive_mean_category_wape).toFixed(2)}% → lightgbm ${(100 * validation.lightgbm_mean_category_wape).toFixed(2)}% mean category wape`}
        />
        <Readout
          label="point-in-time gate"
          value={signedPct(-gate.relative_degradation)}
          title={`${gate.label}: naive ${(100 * gate.naive_mean_category_wape).toFixed(2)}% → lightgbm ${(100 * gate.lightgbm_mean_category_wape).toFixed(2)}% mean category wape — gate FAILED`}
        />
        <Readout label="attribution drift (L1)" value={shift.attribution_drift_l1.toFixed(4)} />
        <Readout
          label="common cutoff / as-of"
          value={`${shift.provenance.common_cutoff} / ${shift.provenance.as_of.slice(0, 10)}`}
        />
      </div>

      <SubHeading>the diagnosis — attribution drift between identical months</SubHeading>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={driftData} layout="vertical" margin={{ top: 4, right: 32, bottom: 0, left: 8 }}>
            <CartesianGrid stroke={GRID} strokeWidth={0.8} horizontal={false} />
            <XAxis
              type="number"
              tick={{ fontSize: 11, fill: MUTED }}
              tickFormatter={(v: number) => (v > 0 ? `+${v.toFixed(3)}` : v.toFixed(3))}
              stroke={GRID}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={118}
              tick={{ fontSize: 11, fill: INK2 }}
              stroke={GRID}
            />
            <ReferenceLine x={0} stroke={MUTED} strokeDasharray="4 3" />
            <Tooltip
              cursor={{ fill: 'rgba(216, 209, 196, 0.25)' }}
              contentStyle={{ background: '#fbf9f4', border: `1px solid ${GRID}`, borderRadius: 2, fontSize: 12 }}
              formatter={(v: number | string, _n: string, entry) => {
                const item = entry?.payload as
                  | { revised: number; point: number; valueShift: number }
                  | undefined;
                if (!item) return [String(v), 'Δ mean |SHAP|'];
                return [
                  `${logDelta(Number(v))} · mean |SHAP| ${item.revised.toFixed(4)} → ${item.point.toFixed(4)} · mean value ${logDelta(item.valueShift)}`,
                  'Δ mean |SHAP|',
                ];
              }}
            />
            <Bar dataKey="drift" isAnimationActive={false} barSize={14} name="Δ mean |SHAP|">
              {driftData.map((row) => (
                <Cell key={row.name} fill={row.drift >= 0 ? POLICY : CRIMSON} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="mt-3 font-mono text-xs lowercase leading-relaxed text-stone-500">
        one architecture, two fits: the same months through {shift.provenance.common_cutoff}, trained
        on latest-revision values (the m3 setting — {(100 * validation.lightgbm_mean_category_wape).toFixed(2)}% mean
        category wape, {(signedPct(validation.relative_improvement))} vs naive) versus point-in-time
        values at the {shift.provenance.as_of.slice(0, 10)} as-of (the gate setting —{' '}
        {(100 * gate.lightgbm_mean_category_wape).toFixed(2)}%, {(signedPct(-gate.relative_degradation))} vs naive,
        refused). the bars are each feature's change in mean |shap| between the two fits: where the
        model's weight moved when history stopped being revisable. the measured endpoints come from
        the frozen fold artifacts the documents cite; nothing here re-runs the gate.
      </p>

      <SubHeading>today's challenger — mean |shap| share over training rows</SubHeading>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={importanceData} layout="vertical" margin={{ top: 4, right: 32, bottom: 0, left: 8 }}>
            <CartesianGrid stroke={GRID} strokeWidth={0.8} horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 'dataMax']}
              tick={{ fontSize: 11, fill: MUTED }}
              tickFormatter={(v: number) => `${v.toFixed(0)}%`}
              stroke={GRID}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={118}
              tick={{ fontSize: 11, fill: INK2 }}
              stroke={GRID}
            />
            <Tooltip
              cursor={{ fill: 'rgba(216, 209, 196, 0.25)' }}
              contentStyle={{ background: '#fbf9f4', border: `1px solid ${GRID}`, borderRadius: 2, fontSize: 12 }}
              formatter={(v: number | string, _n: string, entry) => {
                const item = entry?.payload as { share: number; shap: number } | undefined;
                return [
                  `${Number(v).toFixed(1)}% of mean |SHAP| · mean |SHAP| ${item ? item.shap.toFixed(4) : ''}`,
                  'share',
                ];
              }}
            />
            <Bar dataKey="share" fill={POLICY} isAnimationActive={false} barSize={14}>
              {importanceData.map((row, index) => (
                <Cell key={row.name} fill={index === 0 ? CRIMSON : POLICY} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <SubHeading>one row, decomposed — {provenance.model_status.replace(/_/g, ' ')}</SubHeading>
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <label htmlFor="shap-series" className="font-mono text-[11px] lowercase text-stone-500">
          series
        </label>
        <select
          id="shap-series"
          value={waterfall.series_id}
          onChange={(event) => setSeriesId(event.target.value)}
          className="max-w-full border border-[#d8d1c4] bg-white px-2 py-1 font-mono text-xs text-[#25251f]"
        >
          {waterfalls.map((item) => (
            <option key={item.series_id} value={item.series_id}>
              {item.series_id}
            </option>
          ))}
        </select>
        <span className="font-mono text-[11px] lowercase text-stone-500">
          month {waterfall.month} · model output {logDelta(waterfall.model_output)} log-ratio (
          {pctOf(waterfall.model_output)} vs year-ago)
        </span>
      </div>

      <div className="mt-3 h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={waterfallData} layout="vertical" margin={{ top: 4, right: 32, bottom: 0, left: 8 }}>
            <CartesianGrid stroke={GRID} strokeWidth={0.8} horizontal={false} />
            <XAxis
              type="number"
              domain={waterfallDomain}
              tick={{ fontSize: 11, fill: MUTED }}
              tickFormatter={(v: number) => v.toFixed(2)}
              stroke={GRID}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={128}
              tick={{ fontSize: 11, fill: INK2 }}
              stroke={GRID}
            />
            <ReferenceLine x={waterfall.base_value} stroke={MUTED} strokeDasharray="4 3" />
            <ReferenceLine x={waterfall.model_output} stroke={INK} strokeDasharray="1 3" />
            <Tooltip
              cursor={{ fill: 'rgba(216, 209, 196, 0.25)' }}
              contentStyle={{ background: '#fbf9f4', border: `1px solid ${GRID}`, borderRadius: 2, fontSize: 12 }}
              formatter={(v: number | string, _n: string, entry) => {
                const item = entry?.payload as { shap: number } | undefined;
                if (!item) return [String(v), ''];
                return [
                  `${logDelta(item.shap)} log-ratio (${pctOf(item.shap)} vs year-ago)`,
                  'contribution',
                ];
              }}
            />
            <Bar stackId="waterfall" dataKey="offset" fill="transparent" isAnimationActive={false} />
            <Bar
              stackId="waterfall"
              dataKey="span"
              isAnimationActive={false}
              barSize={16}
              name="contribution"
            >
              {waterfallData.map((row) => (
                <Cell key={row.name} fill={row.sign >= 0 ? CRIMSON : DOWN} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Readout label="model status" value={provenance.model_status.replace(/_/g, ' ')} />
        <Readout label="data cutoff" value={provenance.data_cutoff} />
        <Readout label="training rows" value={provenance.n_training_rows.toLocaleString('en-US')} />
        <Readout
          label="additivity max error"
          value={provenance.additivity_max_error.toExponential(1)}
        />
      </div>

      <p className="mt-3 font-mono text-xs lowercase leading-relaxed text-stone-500">
        mean |shap| share across all {provenance.n_training_rows.toLocaleString('en-US')} training rows ·
        the waterfall decomposes one row (the series' latest training month) from the base{' '}
        {logDelta(waterfall.base_value)} to the output {logDelta(waterfall.model_output)}; the closing
        bar is every feature outside the top {waterfall.contributions.length} · units are the log
        seasonal ratio the model predicts: +0.05 ≈ ~5% above the same month a year earlier · exact
        TreeExplainer values, in-sample, refit {provenance.data_cutoff} gold — an account of the
        challenger's internal arithmetic, not a performance claim.
      </p>
    </div>
  );
};

const Readout: FC<{ label: string; value: string; title?: string }> = ({
  label: readoutLabel,
  value,
  title,
}) => (
  <div className="rounded-sm border border-stone-200 bg-white px-3 py-2" title={title}>
    <p className="font-mono text-[10px] lowercase text-stone-500">{readoutLabel}</p>
    <p className="truncate font-mono text-sm" style={{ color: INK2 }} title={value}>
      {value}
    </p>
  </div>
);
