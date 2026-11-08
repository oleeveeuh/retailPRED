/** τ-sweep slider — drag the policy threshold across the frozen training-window out-of-fold sweep.
 *
 * The REPORT window appears only as the single frozen diamond: sweeping τ on
 * evaluation origins would be re-tuning, so it is never a curve here either.
 */

import { FC, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { policyTauSweep } from '../../content/labData';

const INK = '#25251f';
const INK2 = '#52514e';
const MUTED = '#77776c';
const GRID = '#d8d1c4';
const POLICY = '#89927a';
const CRIMSON = '#a43b36';

const pct = (value: number) => `${(100 * value).toFixed(2)}%`;

export const TauSweepSlider: FC = () => {
  const { sweep, references, report_point: reportPoint, frozen_tau, guardrail_max_loss_pp } =
    policyTauSweep;
  const [tau, setTau] = useState(frozen_tau);

  const nearest = useMemo(() => {
    let best = sweep[0];
    for (const row of sweep) {
      if (Math.abs(row.tau - tau) < Math.abs(best.tau - tau)) best = row;
    }
    return best;
  }, [sweep, tau]);

  const guardrailPasses = nearest.delta_vs_naive_pp <= guardrail_max_loss_pp;
  const data = sweep.map((row) => ({ ...row, y: 100 * row.policy_mean_cat_wape }));
  const guardrailY = 100 * (references.naive + guardrail_max_loss_pp / 100);

  return (
    <div>
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 12, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke={GRID} strokeWidth={0.8} vertical={false} />
            <XAxis
              dataKey="tau"
              domain={[0, 1]}
              type="number"
              tick={{ fontSize: 11, fill: MUTED }}
              tickFormatter={(v: number) => v.toFixed(2)}
              stroke={GRID}
            />
            <YAxis
              domain={[6.9, 14.3]}
              tick={{ fontSize: 11, fill: MUTED }}
              tickFormatter={(v: number) => `${v.toFixed(0)}%`}
              stroke={GRID}
              width={52}
            />
            <Tooltip
              contentStyle={{ background: '#fbf9f4', border: `1px solid ${GRID}`, borderRadius: 2, fontSize: 12 }}
              formatter={(v: number | string, n: string) => [typeof v === 'number' ? pct(v) : v, n]}
              labelFormatter={(l: number) => `τ = ${l.toFixed(2)}`}
            />
            <Line
              type="stepAfter"
              dataKey="y"
              stroke={POLICY}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
              name="switched policy"
            />
            <ReferenceLine y={100 * references.naive} stroke="#1E342E" strokeDasharray="4 3" />
            <ReferenceLine y={100 * references.theta} stroke="#482728" strokeDasharray="4 3" />
            <ReferenceLine y={100 * references.blend} stroke="#b86a62" strokeDasharray="4 3" />
            <ReferenceLine
              y={guardrailY}
              stroke={CRIMSON}
              strokeDasharray="1 3"
              label={{ value: 'guardrail: naive + 0.5pp', position: 'insideBottomRight', fontSize: 10, fill: CRIMSON }}
            />
            <ReferenceDot x={nearest.tau} y={100 * nearest.policy_mean_cat_wape} r={5} fill={POLICY} stroke="#fbf9f4" />
            <ReferenceDot
              x={reportPoint.tau}
              y={100 * reportPoint.policy_mean_cat_wape}
              r={7}
              ifOverflow="extendDomain"
              shape={(props: { cx?: number; cy?: number }) => {
                const cx = props.cx ?? 0;
                const cy = props.cy ?? 0;
                const r = 7;
                return (
                  <path
                    d={`M ${cx} ${cy - r} L ${cx + r} ${cy} L ${cx} ${cy + r} L ${cx - r} ${cy} Z`}
                    fill={INK}
                    stroke="#fbf9f4"
                    strokeWidth={1.5}
                  />
                );
              }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-4 flex items-center gap-4">
        <input
          type="range"
          min={0}
          max={1}
          step={0.02}
          value={tau}
          onChange={(e) => setTau(Number(e.target.value))}
          className="h-1 w-full cursor-pointer accent-[#89927a]"
          aria-label="policy threshold tau"
        />
        <span className="w-14 shrink-0 font-mono text-sm" style={{ color: INK2 }}>
          τ={tau.toFixed(2)}
        </span>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Readout label="theta-call rate" value={`${(100 * nearest.theta_call_rate).toFixed(0)}%`} />
        <Readout label="policy mean-cat WAPE" value={pct(nearest.policy_mean_cat_wape)} />
        <Readout
          label="vs naive on the training window"
          value={`${nearest.delta_vs_naive_pp >= 0 ? '+' : ''}${nearest.delta_vs_naive_pp.toFixed(2)}pp`}
        />
        <Readout label="guardrail 1" value={guardrailPasses ? 'within 0.5pp' : 'violated'} bad={!guardrailPasses} />
      </div>

      <p className="mt-3 font-mono text-xs lowercase text-stone-500">
        curve: training-window out-of-fold (77 pairs) · naive {(100 * references.naive).toFixed(2)}% / theta{' '}
        {(100 * references.theta).toFixed(2)}% / blend {(100 * references.blend).toFixed(2)}% as dashed
        references · the black diamond is the REPORT window at the frozen τ (7.43%, evaluated once —
        never a curve) · frozen τ = {frozen_tau.toFixed(2)}
      </p>
    </div>
  );
};

const Readout: FC<{ label: string; value: string; bad?: boolean }> = ({ label, value, bad }) => (
  <div className="rounded-sm border border-stone-200 bg-white px-3 py-2">
    <p className="font-mono text-[10px] lowercase text-stone-500">{label}</p>
    <p className="font-mono text-sm" style={{ color: bad ? CRIMSON : INK2 }}>
      {value}
    </p>
  </div>
);
