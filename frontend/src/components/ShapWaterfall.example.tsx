/**
 * Example usage of the enhanced ShapWaterfall component
 *
 * This demonstrates how to use the publication-quality SHAP visualization
 * with all its interactive features.
 */

import React from 'react';
import { ShapWaterfall, SHAPWaterfallData } from './ShapWaterfall';

export const ShapWaterfallExample: React.FC = () => {
  // Example data for a retail sales prediction
  const exampleData: SHAPWaterfallData[] = [
    {
      feature: 'Lag_1 (Previous Month)',
      value: 15234.56,
      contribution: 15234.56,
      isPositive: true,
      importance: 35.8,
      historical: [12000, 13500, 14200, 15100, 14900, 15234.56],
      distribution: { min: 8000, max: 18000, mean: 13000, std: 2500 },
      correlation: 0.87,
    },
    {
      feature: 'Unemployment Rate',
      value: -8945.23,
      contribution: 6289.33,
      isPositive: false,
      importance: 22.4,
      historical: [-5000, -6500, -7200, -8100, -8500, -8945.23],
      distribution: { min: -12000, max: -2000, mean: -7000, std: 2800 },
      correlation: -0.76,
    },
    {
      feature: 'Consumer Confidence',
      value: 5678.90,
      contribution: 11968.23,
      isPositive: true,
      importance: 18.2,
      historical: [3000, 4200, 4800, 5100, 5400, 5678.90],
      distribution: { min: 1000, max: 8000, mean: 4500, std: 1800 },
      correlation: 0.69,
    },
    {
      feature: 'Seasonal_December',
      value: 4321.45,
      contribution: 16289.68,
      isPositive: true,
      importance: 12.6,
      historical: [2000, 2500, 3000, 3800, 4100, 4321.45],
      distribution: { min: 500, max: 6000, mean: 3200, std: 1500 },
      correlation: 0.58,
    },
    {
      feature: 'Interest Rate',
      value: -2345.67,
      contribution: 13944.01,
      isPositive: false,
      importance: 8.9,
      historical: [-1000, -1500, -1800, -2100, -2200, -2345.67],
      distribution: { min: -4000, max: -500, mean: -2000, std: 900 },
      correlation: -0.45,
    },
    {
      feature: 'Gasoline Price',
      value: -1234.89,
      contribution: 12709.12,
      isPositive: false,
      importance: 2.1,
      historical: [-500, -800, -1000, -1100, -1180, -1234.89],
      distribution: { min: -2000, max: -200, mean: -1000, std: 500 },
      correlation: -0.32,
    },
  ];

  const baseValue = 659843.45;
  const finalValue = 672552.57;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 p-4 sm:p-8">
      <div className="max-w-7xl mx-auto">
        {/* Page Header */}
        <div className="mb-8">
          <h1 className="text-3xl sm:text-4xl font-bold text-slate-900 mb-2">
            SHAP Value Analysis
          </h1>
          <p className="text-slate-600 text-lg">
            Interactive feature contribution breakdown for model predictions
          </p>
        </div>

        {/* Main ShapWaterfall Component */}
        <ShapWaterfall
          data={exampleData}
          baseValue={baseValue}
          finalValue={finalValue}
          title="December 2025 Sales Prediction"
          categoryName="Total Retail Sales"
          predictionId={42}
          height={600}
          showValues={true}
        />

        {/* Usage Notes */}
        <div className="mt-8 bg-white rounded-2xl shadow-lg border border-slate-200 p-6">
          <h2 className="text-xl font-bold text-slate-900 mb-4">
            Component Features
          </h2>
          <div className="grid md:grid-cols-2 gap-6 text-sm text-slate-700">
            <div>
              <h3 className="font-semibold text-slate-900 mb-2">
                🎨 Three View Modes
              </h3>
              <ul className="space-y-1">
                <li>• <strong>Waterfall:</strong> Horizontal bars showing cumulative flow</li>
                <li>• <strong>Force Plot:</strong> Animated progress bars with gradients</li>
                <li>• <strong>Beeswarm:</strong> Scatter plot of SHAP values</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-slate-900 mb-2">
                🖱️ Interactive Elements
              </h3>
              <ul className="space-y-1">
                <li>• Click any feature bar for deep-dive analysis</li>
                <li>• Hover over bars for detailed tooltips</li>
                <li>• Export chart as high-resolution PNG</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-slate-900 mb-2">
                📊 Feature Deep-Dive
              </h3>
              <ul className="space-y-1">
                <li>• Historical importance trend (6 months)</li>
                <li>• Value distribution with visual indicator</li>
                <li>• Correlation gauge with outcomes</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-slate-900 mb-2">
                ✨ Design Quality
              </h3>
              <ul className="space-y-1">
                <li>• Professional color palette (emerald/red/blue)</li>
                <li>• Smooth animations with Framer Motion</li>
                <li>• Fully responsive for mobile/tablet/desktop</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Code Example */}
        <div className="mt-6 bg-slate-900 rounded-2xl p-6 overflow-x-auto">
          <h3 className="text-lg font-semibold text-white mb-4">Usage Example</h3>
          <pre className="text-sm text-slate-300">
{`import { ShapWaterfall, SHAPWaterfallData } from './ShapWaterfall';

const data: SHAPWaterfallData[] = [
  {
    feature: 'Lag_1 (Previous Month)',
    value: 15234.56,
    contribution: 15234.56,
    isPositive: true,
    importance: 35.8,
    historical: [12000, 13500, 14200, 15100, 14900, 15234],
    distribution: { min: 8000, max: 18000, mean: 13000, std: 2500 },
    correlation: 0.87,
  },
  // ... more features
];

<ShapWaterfall
  data={data}
  baseValue={659843.45}
  finalValue={672552.57}
  title="December 2025 Sales Prediction"
  categoryName="Total Retail Sales"
  predictionId={42}
  height={600}
  showValues={true}
/>`}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default ShapWaterfallExample;
