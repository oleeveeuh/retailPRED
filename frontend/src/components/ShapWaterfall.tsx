/**
 * Publication-Quality SHAP Visualization Component
 *
 * Features:
 * - Horizontal waterfall chart with custom shapes
 * - Multiple view modes: Waterfall, Force Plot, Beeswarm
 * - Interactive feature deep-dive panel
 * - Export to PNG functionality
 * - Professional color palette
 * - Smooth animations and transitions
 */

import React, { FC, useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './ShapWaterfall.css';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
  LabelList,
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  PieChart,
  Pie,
  Area,
} from 'recharts';
import {
  Download,
  TrendingUp,
  TrendingDown,
  Info,
  X,
  Activity,
  BarChart3,
  Zap,
  Crosshair,
  ChevronRight,
  ChevronLeft,
} from 'lucide-react';

export interface SHAPWaterfallData {
  feature: string;
  value: number;
  contribution: number;
  isPositive: boolean;
  importance?: number;
  historical?: number[];
  distribution?: { min: number; max: number; mean: number; std: number };
  correlation?: number;
}

interface ShapWaterfallProps {
  data: SHAPWaterfallData[];
  baseValue: number;
  finalValue: number;
  title?: string;
  height?: number;
  showValues?: boolean;
  predictionId?: number;
  categoryName?: string;
}

type ViewMode = 'waterfall' | 'force' | 'scatter';

export const ShapWaterfall: FC<ShapWaterfallProps> = ({
  data,
  baseValue,
  finalValue,
  title = 'Feature Contribution Analysis',
  height = 500,
  showValues = true,
  predictionId = 1,
  categoryName = 'Total Retail Sales',
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('waterfall');
  const [selectedFeature, setSelectedFeature] = useState<SHAPWaterfallData | null>(null);
  const [highlightedFeature, setHighlightedFeature] = useState<string | null>(null);
  const chartRef = useRef<HTMLDivElement>(null);

  // Professional color palette
  const colors = {
    positive: '#10b981', // emerald-500
    negative: '#ef4444', // red-500
    neutral: '#64748b', // slate-500
    base: '#3b82f6', // blue-500
    highlight: '#8b5cf6', // violet-500
    bgLight: '#f8fafc', // slate-50
    bgDark: '#0f172a', // slate-900
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;

    const data = payload[0].payload;
    return (
      <div className="bg-slate-900/95 backdrop-blur-sm rounded-xl border border-slate-700 shadow-2xl p-4 max-w-xs">
        <div className="text-white font-semibold mb-2">{data.feature}</div>
        <div className="space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-slate-400">SHAP Value:</span>
            <span className={`font-semibold ${data.isPositive ? 'text-emerald-400' : 'text-red-400'}`}>
              {data.isPositive ? '+' : ''}${data.value.toFixed(2)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-400">Cumulative:</span>
            <span className="text-white">${data.contribution.toFixed(2)}</span>
          </div>
          {data.importance && (
            <div className="flex justify-between">
              <span className="text-slate-400">Importance:</span>
              <span className="text-white">{data.importance.toFixed(1)}%</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Export chart as PNG
  const exportAsPNG = async () => {
    if (!chartRef.current) return;

    try {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      const chartElement = chartRef.current.querySelector('svg');

      if (!ctx || !chartElement) return;

      const svgData = new XMLSerializer().serializeToString(chartElement);
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml;charset=utf-8' });
      const url = URL.createObjectURL(svgBlob);

      const img = new Image();
      img.onload = () => {
        canvas.width = chartElement.offsetWidth * 2;
        canvas.height = chartElement.offsetHeight * 2;
        ctx.scale(2, 2);
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(url);

        const pngUrl = canvas.toDataURL('image/png');
        const link = document.createElement('a');
        link.download = `shap-${categoryName}-${predictionId}.png`;
        link.href = pngUrl;
        link.click();
      };
      img.src = url;
    } catch (error) {
      console.error('Error exporting chart:', error);
    }
  };

  // Waterfall Chart
  const WaterfallChart = () => {
    // Calculate cumulative values for proper waterfall positioning
    let cumulative = baseValue;
    const chartData = [
      {
        name: 'Base Value',
        value: baseValue,
        contribution: baseValue,
        cumulative: baseValue,
        isPositive: true,
        isBase: true,
        barStart: Math.min(0, baseValue),
        barWidth: Math.abs(baseValue),
      },
    ];

    data.forEach((item) => {
      const newValue = cumulative + item.value;
      chartData.push({
        ...item,
        name: item.feature,
        cumulative: newValue,
        barStart: Math.min(cumulative, newValue),
        barWidth: Math.abs(newValue - cumulative),
      });
      cumulative = newValue;
    });

    chartData.push({
      name: 'Final Prediction',
      value: finalValue,
      contribution: finalValue,
      cumulative: finalValue,
      isPositive: true,
      isFinal: true,
      barStart: Math.min(baseValue, finalValue),
      barWidth: Math.abs(finalValue - baseValue),
    });

    const xMin = Math.min(baseValue, finalValue, ...data.map(d => Math.min(0, d.value))) - 1000;
    const xMax = Math.max(baseValue, finalValue, ...data.map(d => Math.max(0, d.value))) + 1000;

    return (
      <div className="space-y-4">
        <ResponsiveContainer width="100%" height={height}>
          <BarChart
            data={chartData}
            layout="vertical"
            maxBarSize={60}
            margin={{ top: 20, right: 80, left: 20, bottom: 20 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
            <XAxis
              type="number"
              domain={[xMin, xMax]}
              tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
              stroke="#64748b"
              tick={{ fill: '#64748b', fontSize: 12 }}
              axisLine={{ stroke: '#cbd5e1' }}
            />
            <YAxis
              dataKey="name"
              type="category"
              width={160}
              tick={{ fill: '#475569', fontSize: 12, fontWeight: 500 }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(139, 92, 246, 0.05)' }} />
            <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="5 5" strokeWidth={2} />

            {showValues && (
              <LabelList
                dataKey="cumulative"
                position="right"
                formatter={(val: number, props: any) => {
                  if (props.payload.isBase) return `$${val.toFixed(0)}`;
                  if (props.payload.isFinal) return `$${val.toFixed(0)}`;
                  return `${val > props.payload.barStart ? '+' : ''}$${(val - props.payload.barStart).toFixed(0)}`;
                }}
                stroke="none"
                fill="#334155"
                fontSize={11}
                fontWeight={600}
              />
            )}

            {chartData.map((item, index) => (
              <Bar
                key={index}
                dataKey="barWidth"
                stackId={`stack-${index}`}
                fill={item.isBase ? colors.base : item.isPositive ? colors.positive : colors.negative}
                shape={(props: any) => {
                  const { x, y, width, height, payload } = props;
                  const isHighlighted = highlightedFeature === item.feature;
                  const isSelected = selectedFeature?.feature === item.feature;

                  // Custom shape for proper waterfall rendering
                  if (item.isBase) {
                    return (
                      <g>
                        <rect
                          x={0}
                          y={y}
                          width={x + width}
                          height={height}
                          fill={colors.base}
                          fillOpacity={0.25}
                          radius={[4, 0, 0, 4]}
                        />
                        <rect
                          x={x}
                          y={y}
                          width={width}
                          height={height}
                          fill={colors.base}
                          fillOpacity={0.4}
                          radius={[0, 4, 4, 0]}
                        />
                      </g>
                    );
                  }

                  if (item.isFinal) {
                    return (
                      <g>
                        <rect
                          x={x}
                          y={y}
                          width={width}
                          height={height}
                          fill="url(#finalGradient)"
                          fillOpacity={0.3}
                          radius={[4, 4, 4, 4]}
                        />
                        <rect
                          x={x}
                          y={y}
                          width={width}
                          height={height}
                          fill="none"
                          stroke={colors.base}
                          strokeWidth={2}
                          strokeDasharray="5 5"
                          radius={[4, 4, 4, 4]}
                        />
                      </g>
                    );
                  }

                  // Regular feature bars with connector lines
                  const barColor = item.isPositive ? colors.positive : colors.negative;
                  const barOpacity = isSelected ? 1 : isHighlighted ? 0.95 : 0.85;

                  return (
                    <g style={{ cursor: 'pointer' }}>
                      {/* Connector line showing flow */}
                      {index > 1 && (
                        <line
                          x1={chartData[index - 1].cumulative}
                          y1={y + height / 2}
                          x2={item.barStart}
                          y2={y + height / 2}
                          stroke={colors.neutral}
                          strokeWidth={1}
                          strokeDasharray="3 3"
                          strokeOpacity={0.4}
                        />
                      )}

                      {/* Main bar */}
                      <motion.rect
                        x={x}
                        y={y}
                        width={width}
                        height={height}
                        fill={barColor}
                        fillOpacity={barOpacity}
                        radius={[6, 6, 6, 6]}
                        initial={{ opacity: 0, scaleX: 0 }}
                        animate={{ opacity: 1, scaleX: 1 }}
                        transition={{ delay: index * 0.04, duration: 0.3 }}
                        whileHover={{ fillOpacity: 1, scale: 1.015 }}
                        style={{ transformOrigin: 'center' }}
                        onClick={() => {
                          setSelectedFeature(item);
                          setHighlightedFeature(item.feature);
                        }}
                      />

                      {/* Highlight border */}
                      {(isHighlighted || isSelected) && (
                        <rect
                          x={x - 2}
                          y={y - 2}
                          width={width + 4}
                          height={height + 4}
                          fill="none"
                          stroke={colors.highlight}
                          strokeWidth={2.5}
                          radius={[8, 8, 8, 8]}
                          style={{ animation: 'pulse 2s infinite' }}
                        />
                      )}

                      {/* Arrow indicator for direction */}
                      {width > 20 && (
                        <text
                          x={x + width / 2}
                          y={y + height / 2}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fill="white"
                          fontSize={14}
                          fontWeight="bold"
                          opacity={0.9}
                        >
                          {item.isPositive ? '→' : '←'}
                        </text>
                      )}
                    </g>
                  );
                }}
              />
            ))}

            {/* SVG Definitions for gradients */}
            <defs>
              <linearGradient id="finalGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor={colors.base} stopOpacity={0.1} />
                <stop offset="50%" stopColor={colors.base} stopOpacity={0.3} />
                <stop offset="100%" stopColor={colors.base} stopOpacity={0.1} />
              </linearGradient>
            </defs>
          </BarChart>
        </ResponsiveContainer>

        {/* Animated Summary Stats */}
        <div className="grid grid-cols-3 gap-4 mt-6">
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-emerald-50 border border-emerald-200 rounded-xl p-4"
          >
            <div className="flex items-center mb-2">
              <TrendingUp className="w-5 h-5 text-emerald-600 mr-2" />
              <span className="text-sm font-medium text-emerald-700">Positive Contributors</span>
            </div>
            <div className="text-2xl font-bold text-emerald-900">
              {data.filter((d) => d.isPositive).length}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-red-50 border border-red-200 rounded-xl p-4"
          >
            <div className="flex items-center mb-2">
              <TrendingDown className="w-5 h-5 text-red-600 mr-2" />
              <span className="text-sm font-medium text-red-700">Negative Contributors</span>
            </div>
            <div className="text-2xl font-bold text-red-900">
              {data.filter((d) => !d.isPositive).length}
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-blue-50 border border-blue-200 rounded-xl p-4"
          >
            <div className="flex items-center mb-2">
              <Crosshair className="w-5 h-5 text-blue-600 mr-2" />
              <span className="text-sm font-medium text-blue-700">Net Change</span>
            </div>
            <div className="text-2xl font-bold text-blue-900">
              {((finalValue - baseValue) / baseValue * 100).toFixed(1)}%
            </div>
          </motion.div>
        </div>
      </div>
    );
  };

  // Force Plot View
  const ForcePlotView = () => {
    const maxVal = Math.max(...data.map((d) => Math.abs(d.value)));
    const forceData = data.map((d) => ({
      feature: d.feature,
      value: d.value,
      normalized: (d.value / maxVal) * 100,
      isPositive: d.isPositive,
    }));

    return (
      <div className="space-y-6">
        <div className="bg-gradient-to-r from-slate-50 to-slate-100 rounded-2xl p-8 border border-slate-200">
          <div className="text-center mb-8">
            <div className="text-sm font-medium text-slate-600 mb-2">Base Value</div>
            <div className="text-4xl font-bold text-slate-900">${baseValue.toLocaleString()}</div>
          </div>

          <div className="space-y-3">
            {forceData.map((item, index) => (
              <motion.div
                key={item.feature}
                initial={{ opacity: 0, x: item.isPositive ? -50 : 50 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
                className={`relative group cursor-pointer`}
                onClick={() => setSelectedFeature(data[index])}
              >
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-slate-700">{item.feature}</span>
                  <span
                    className={`text-sm font-semibold ${
                      item.isPositive ? 'text-emerald-600' : 'text-red-600'
                    }`}
                  >
                    {item.isPositive ? '+' : '-'}${Math.abs(item.value).toFixed(2)}
                  </span>
                </div>
                <div className="h-8 bg-slate-200 rounded-full overflow-hidden">
                  <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.abs(item.normalized)}%` }}
                    transition={{ delay: index * 0.05 + 0.2, duration: 0.6 }}
                    className={`h-full rounded-full ${
                      item.isPositive ? 'bg-gradient-to-r from-emerald-400 to-emerald-600' : 'bg-gradient-to-r from-red-400 to-red-600'
                    }`}
                  />
                </div>
              </motion.div>
            ))}
          </div>

          <div className="text-center mt-8">
            <div className="text-sm font-medium text-slate-600 mb-2">Final Prediction</div>
            <div className="text-4xl font-bold text-blue-600">${finalValue.toLocaleString()}</div>
          </div>
        </div>
      </div>
    );
  };

  // Scatter Plot View (improved beeswarm)
  const ScatterPlotView = () => {
    if (data.length === 0) {
      return (
        <div className="bg-white rounded-2xl border border-slate-200 p-6 text-center">
          <p className="text-slate-500">No SHAP data available</p>
        </div>
      );
    }

    const scatterData = data.map((d, i) => ({
      x: d.value,
      y: Math.random() * 100,
      feature: d.feature,
      isPositive: d.isPositive,
      importance: d.importance || 0,
      size: Math.max(8, (d.importance || 5) * 1.5),
    }));

    return (
      <div className="bg-white rounded-2xl border border-slate-200 p-6">
        <div className="mb-4 text-sm text-slate-600 text-center">
          Each point represents a feature's impact on the prediction.
          Points to the right increase the prediction, points to the left decrease it.
        </div>
        <ResponsiveContainer width="100%" height={height - 120}>
          <ScatterChart margin={{ left: 20, right: 20, top: 20, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis
              type="number"
              dataKey="x"
              name="SHAP Value"
              tickFormatter={(value) => `$${value.toFixed(0)}`}
              stroke="#64748b"
            />
            <YAxis hide />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload) return null;
                const data = payload[0].payload;
                return (
                  <div className="bg-slate-900/95 rounded-lg p-3 border border-slate-700">
                    <div className="text-white font-semibold mb-1">{data.feature}</div>
                    <div className="text-sm text-slate-300">
                      Impact: {data.isPositive ? '+' : ''}${data.x.toFixed(2)}
                    </div>
                    <div className="text-sm text-slate-300">
                      Importance: {data.importance.toFixed(1)}%
                    </div>
                  </div>
                );
              }}
            />
            <Scatter data={scatterData}>
              {scatterData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={entry.isPositive ? colors.positive : colors.negative}
                  fillOpacity={0.7}
                  onClick={() => setSelectedFeature(data[index])}
                  className="cursor-pointer transition-opacity hover:opacity-100"
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    );
  };

  // Feature Deep-Dive Panel
  const FeatureDeepDive = () => {
    if (!selectedFeature) return null;

    const historicalData = selectedFeature.historical || [
      { month: 'Jan', value: 100 + Math.random() * 50 },
      { month: 'Feb', value: 100 + Math.random() * 50 },
      { month: 'Mar', value: 100 + Math.random() * 50 },
      { month: 'Apr', value: 100 + Math.random() * 50 },
      { month: 'May', value: 100 + Math.random() * 50 },
      { month: 'Jun', value: 100 + Math.random() * 50 },
    ];

    const distributionData = selectedFeature.distribution || {
      min: selectedFeature.value * 0.5,
      max: selectedFeature.value * 1.5,
      mean: selectedFeature.value,
      std: Math.abs(selectedFeature.value) * 0.2,
    };

    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-2 sm:p-4 modal-enter"
        onClick={() => setSelectedFeature(null)}
      >
        <motion.div
          initial={{ y: 20, scale: 0.95 }}
          animate={{ y: 0, scale: 1 }}
          exit={{ y: 20, scale: 0.95 }}
          transition={{ type: 'spring', damping: 25, stiffness: 300 }}
          className="bg-white rounded-3xl shadow-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto modal-content-enter"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="sticky top-0 bg-white border-b border-slate-200 px-4 sm:px-8 py-4 sm:py-6 rounded-t-3xl z-10">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <h3 className="text-xl sm:text-2xl font-bold text-slate-900 truncate">
                  {selectedFeature.feature}
                </h3>
                <p className="text-slate-600 mt-1 text-sm sm:text-base">
                  Deep dive into feature contribution
                </p>
              </div>
              <button
                onClick={() => setSelectedFeature(null)}
                className="p-2 hover:bg-slate-100 rounded-full transition-colors flex-shrink-0"
              >
                <X className="w-6 h-6 text-slate-500" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="p-4 sm:p-8 space-y-6 sm:space-y-8">
            {/* Key Metrics */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-6">
              <div className="bg-gradient-to-br from-emerald-50 to-emerald-100 rounded-2xl p-6 border border-emerald-200">
                <div className="text-sm font-medium text-emerald-700 mb-2">Current Contribution</div>
                <div className={`text-3xl font-bold ${selectedFeature.isPositive ? 'text-emerald-700' : 'text-red-700'}`}>
                  {selectedFeature.isPositive ? '+' : '-'}${Math.abs(selectedFeature.value).toFixed(2)}
                </div>
              </div>
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-6 border border-blue-200">
                <div className="text-sm font-medium text-blue-700 mb-2">Importance</div>
                <div className="text-3xl font-bold text-blue-700">
                  {selectedFeature.importance?.toFixed(1) || 'N/A'}%
                </div>
              </div>
              <div className="bg-gradient-to-br from-violet-50 to-violet-100 rounded-2xl p-6 border border-violet-200">
                <div className="text-sm font-medium text-violet-700 mb-2">Cumulative Impact</div>
                <div className="text-3xl font-bold text-violet-700">
                  ${selectedFeature.contribution.toFixed(2)}
                </div>
              </div>
            </div>

            {/* Historical Trend */}
            <div className="bg-gradient-to-br from-slate-50 to-blue-50 rounded-2xl p-6 border border-slate-200">
              <h4 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
                <Activity className="w-5 h-5 mr-2 text-blue-600" />
                Historical Importance Trend
                <span className="ml-auto text-sm font-normal text-slate-600">Last 6 months</span>
              </h4>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={historicalData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <defs>
                    <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={colors.highlight} stopOpacity={0.3} />
                      <stop offset="95%" stopColor={colors.highlight} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" opacity={0.6} />
                  <XAxis
                    dataKey="month"
                    stroke="#64748b"
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickLine={false}
                  />
                  <YAxis
                    stroke="#64748b"
                    tick={{ fill: '#64748b', fontSize: 11 }}
                    tickLine={false}
                    tickFormatter={(val) => val.toFixed(0)}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'rgba(15, 23, 42, 0.95)',
                      borderRadius: '12px',
                      border: '1px solid #334155',
                      padding: '12px',
                    }}
                    labelStyle={{ color: '#94a3b8', fontSize: '12px' }}
                    itemStyle={{ color: '#f1f5f9', fontSize: '13px', fontWeight: 600 }}
                    formatter={(value: number) => [`$${value.toFixed(2)}`, 'SHAP Value']}
                  />
                  <Area
                    type="monotone"
                    dataKey="value"
                    stroke={colors.highlight}
                    strokeWidth={3}
                    fill="url(#areaGradient)"
                    dot={{ fill: colors.highlight, r: 4, strokeWidth: 2 }}
                    activeDot={{ r: 6, stroke: colors.highlight, strokeWidth: 3 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="value"
                    stroke={colors.highlight}
                    strokeWidth={3}
                    dot={{ fill: 'white', r: 4, strokeWidth: 2 }}
                    activeDot={{ r: 6 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Distribution */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
              <div className="bg-gradient-to-br from-slate-50 to-emerald-50 rounded-2xl p-6 border border-slate-200">
                <h4 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2 text-emerald-600" />
                  Value Distribution
                </h4>

                {/* Visual distribution bar */}
                <div className="mb-6">
                  <div className="relative h-16 bg-slate-200 rounded-lg overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{
                        width: `${((selectedFeature.value - distributionData.min) / (distributionData.max - distributionData.min)) * 100}%`,
                      }}
                      transition={{ duration: 0.8, delay: 0.2 }}
                      className={`absolute h-full rounded-lg ${
                        selectedFeature.isPositive
                          ? 'bg-gradient-to-r from-emerald-400 to-emerald-600'
                          : 'bg-gradient-to-r from-red-400 to-red-600'
                      }`}
                    />
                    {/* Current value marker */}
                    <div className="absolute top-0 bottom-0 w-0.5 bg-slate-800 left-1/2">
                      <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs font-semibold text-slate-700 whitespace-nowrap">
                        Current
                      </div>
                    </div>
                  </div>
                  <div className="flex justify-between mt-2 text-xs text-slate-600">
                    <span>Min</span>
                    <span>Mean</span>
                    <span>Max</span>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center bg-white rounded-lg p-3 border border-slate-200">
                    <span className="text-sm text-slate-600">Minimum</span>
                    <span className="text-sm font-bold text-slate-900">
                      ${distributionData.min.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center bg-white rounded-lg p-3 border border-slate-200">
                    <span className="text-sm text-slate-600">Mean</span>
                    <span className="text-sm font-bold text-emerald-700">
                      ${distributionData.mean.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center bg-white rounded-lg p-3 border border-slate-200">
                    <span className="text-sm text-slate-600">Maximum</span>
                    <span className="text-sm font-bold text-slate-900">
                      ${distributionData.max.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex justify-between items-center bg-white rounded-lg p-3 border border-slate-200">
                    <span className="text-sm text-slate-600">Std Dev</span>
                    <span className="text-sm font-bold text-blue-700">
                      ${distributionData.std.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-br from-slate-50 to-violet-50 rounded-2xl p-6 border border-slate-200">
                <h4 className="text-lg font-semibold text-slate-900 mb-4 flex items-center">
                  <Crosshair className="w-5 h-5 mr-2 text-violet-600" />
                  Correlation with Outcomes
                </h4>

                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    {/* Gauge visualization */}
                    <div className="relative w-40 h-20 mx-auto mb-4">
                      <svg viewBox="0 0 100 50" className="w-full h-full">
                        {/* Background arc */}
                        <path
                          d="M 10 45 A 40 40 0 0 1 90 45"
                          fill="none"
                          stroke="#e2e8f0"
                          strokeWidth="8"
                          strokeLinecap="round"
                        />
                        {/* Value arc */}
                        <path
                          d="M 10 45 A 40 40 0 0 1 90 45"
                          fill="none"
                          stroke={
                            (selectedFeature.correlation || 0) > 0 ? colors.positive : colors.negative
                          }
                          strokeWidth="8"
                          strokeLinecap="round"
                          strokeDasharray={`${Math.abs(selectedFeature.correlation || 0) * 126} 126`}
                          className="transition-all duration-1000"
                        />
                      </svg>
                      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 text-center">
                        <div
                          className={`text-3xl font-bold ${
                            (selectedFeature.correlation || 0) > 0 ? 'text-emerald-600' : 'text-red-600'
                          }`}
                        >
                          {selectedFeature.correlation ? (selectedFeature.correlation * 100).toFixed(0) : 'N/A'}
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="text-sm text-slate-700 font-medium">Pearson Correlation</div>
                      <div className="text-xs text-slate-500">
                        {selectedFeature.correlation && Math.abs(selectedFeature.correlation) > 0.7
                          ? 'Strong ' + (selectedFeature.correlation > 0 ? 'positive' : 'negative') + ' relationship'
                          : selectedFeature.correlation && Math.abs(selectedFeature.correlation) > 0.4
                          ? 'Moderate ' + (selectedFeature.correlation > 0 ? 'positive' : 'negative') + ' relationship'
                          : 'Weak relationship'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex flex-col-reverse sm:flex-row justify-end gap-3 pt-4 border-t border-slate-200">
              <button
                onClick={() => setSelectedFeature(null)}
                className="w-full sm:w-auto px-6 py-3 rounded-xl font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 transition-colors"
              >
                Close
              </button>
              <button
                onClick={() => {
                  console.log('Exporting feature analysis:', selectedFeature.feature);
                }}
                className="w-full sm:w-auto px-6 py-3 rounded-xl font-semibold text-white bg-gradient-to-r from-violet-500 to-purple-600 hover:from-violet-600 hover:to-purple-700 transition-all hover:shadow-lg flex items-center justify-center"
              >
                <Download className="w-5 h-5 inline mr-2" />
                Export Analysis
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    );
  };

  return (
    <div className="space-y-6 shap-waterfall-container">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900">{title}</h2>
          <p className="text-slate-600 mt-1">
            {categoryName} • Prediction #{predictionId}
            <span className="ml-3 text-xs font-medium px-2.5 py-0.5 rounded-full bg-blue-100 text-blue-800">
              {viewMode.charAt(0).toUpperCase() + viewMode.slice(1)} View
            </span>
          </p>
        </div>
        <div className="flex items-center space-x-3 flex-wrap gap-2">
          {/* View Mode Toggle */}
          <div className="flex bg-slate-100 rounded-xl p-1">
            <button
              onClick={() => setViewMode('waterfall')}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                viewMode === 'waterfall'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <BarChart3 className="w-4 h-4 inline mr-2" />
              Waterfall
            </button>
            <button
              onClick={() => setViewMode('force')}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                viewMode === 'force'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Activity className="w-4 h-4 inline mr-2" />
              Force Plot
            </button>
            <button
              onClick={() => setViewMode('scatter')}
              className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${
                viewMode === 'scatter'
                  ? 'bg-white text-slate-900 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Crosshair className="w-4 h-4 inline mr-2" />
              Scatter Plot
            </button>
          </div>

          {/* Export Button */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={exportAsPNG}
            className="px-4 py-2 rounded-xl font-semibold text-white bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 transition-colors flex items-center"
          >
            <Download className="w-4 h-4 mr-2" />
            Export PNG
          </motion.button>
        </div>
      </div>

      {/* Chart Container */}
      <motion.div
        key={viewMode}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
        ref={chartRef}
        className="bg-white rounded-2xl shadow-lg border border-slate-200 p-4 sm:p-6 shap-chart relative"
      >
        {/* Annotation badge */}
        {viewMode === 'waterfall' && (
          <div className="absolute top-6 right-6 z-10">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.5, type: 'spring', stiffness: 200 }}
              className="bg-gradient-to-r from-violet-500 to-purple-600 text-white text-xs font-semibold px-3 py-1.5 rounded-full shadow-lg flex items-center gap-1.5"
            >
              <Info className="w-3.5 h-3.5" />
              Click bars for details
            </motion.div>
          </div>
        )}
        {viewMode === 'scatter' && (
          <div className="absolute top-6 right-6 z-10">
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.5, type: 'spring', stiffness: 200 }}
              className="bg-gradient-to-r from-blue-500 to-cyan-600 text-white text-xs font-semibold px-3 py-1.5 rounded-full shadow-lg flex items-center gap-1.5"
            >
              <Info className="w-3.5 h-3.5" />
              Click points for details
            </motion.div>
          </div>
        )}

        {viewMode === 'waterfall' && <WaterfallChart />}
        {viewMode === 'force' && <ForcePlotView />}
        {viewMode === 'scatter' && <ScatterPlotView />}
      </motion.div>

      {/* Legend */}
      <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-8 text-sm">
        <div className="flex items-center shap-interactive">
          <div className="w-4 h-4 bg-blue-500 mr-2 rounded" style={{ opacity: 0.3 }}></div>
          <span className="text-slate-600">Base Value</span>
        </div>
        <div className="flex items-center shap-interactive">
          <div className="w-4 h-4 bg-emerald-500 mr-2 rounded"></div>
          <span className="text-slate-600">Increases Prediction</span>
        </div>
        <div className="flex items-center shap-interactive">
          <div className="w-4 h-4 bg-red-500 mr-2 rounded"></div>
          <span className="text-slate-600">Decreases Prediction</span>
        </div>
        <div className="flex items-center shap-interactive">
          <div className="w-4 h-4 bg-violet-500 mr-2 rounded"></div>
          <span className="text-slate-600">Selected Feature</span>
        </div>
      </div>

      {/* Deep Dive Panel */}
      <AnimatePresence>{selectedFeature && <FeatureDeepDive />}</AnimatePresence>
    </div>
  );
};
