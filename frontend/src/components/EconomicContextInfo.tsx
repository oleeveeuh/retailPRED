/**
 * EconomicContextInfo Component
 *
 * Educational panel explaining how economic context works in RetailPRED.
 * Clearly distinguishes between prediction (time-series features) and interpretation (economic data).
 */

import React, { useState } from 'react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { ChevronDown, ChevronUp } from 'lucide-react'

export function EconomicContextInfo() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <div className="mt-4">
      <Button
        variant="outline"
        size="sm"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full justify-between"
      >
        <span>ℹ️ About Economic Context</span>
        {isOpen ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
      </Button>

      {isOpen && (
        <Alert className="mt-3 bg-blue-50 border-blue-200">
          <AlertDescription>
            <div className="space-y-3">
              <div>
                <h4 className="font-semibold text-blue-900 mb-2">
                  How Economic Context Works
                </h4>

                <div className="space-y-3 text-sm text-gray-700">
                  <div className="bg-white rounded p-3 border border-blue-100">
                    <p className="font-medium text-blue-900 mb-1">
                      📊 Model Predictions (0.26% MAPE)
                    </p>
                    <p className="text-gray-700">
                      Our model uses <strong>74 time-series features</strong> from retail sales data:
                    </p>
                    <ul className="list-disc list-inside ml-4 mt-2 space-y-1 text-gray-600">
                      <li>Recent sales values (lags)</li>
                      <li>Rolling statistics (means, standard deviations)</li>
                      <li>Seasonal patterns (month, quarter)</li>
                      <li>Trend indicators (momentum, acceleration)</li>
                    </ul>
                    <p className="text-xs text-gray-500 mt-2">
                      ✅ Economic indicators are <strong>NOT used</strong> for predictions
                    </p>
                  </div>

                  <div className="bg-white rounded p-3 border border-blue-100">
                    <p className="font-medium text-blue-900 mb-1">
                      💡 Economic Context (Interpretation Only)
                    </p>
                    <p className="text-gray-700">
                      We overlay economic indicators to help <strong>explain</strong> predictions:
                    </p>
                    <ul className="list-disc list-inside ml-4 mt-2 space-y-1 text-gray-600">
                      <li>Unemployment rate (from FRED)</li>
                      <li>Consumer confidence index</li>
                      <li>Federal funds rate</li>
                      <li>Anomaly detection (z-scores)</li>
                    </ul>
                    <p className="text-xs text-gray-500 mt-2">
                      ✅ Used for <strong>interpretation only</strong>, not prediction
                    </p>
                  </div>

                  <div className="bg-white rounded p-3 border border-blue-100">
                    <p className="font-medium text-blue-900 mb-2">
                      🎯 Use Cases
                    </p>
                    <ul className="list-disc list-inside ml-4 space-y-1 text-gray-600">
                      <li>Understanding historical anomalies (e.g., COVID-19, recessions)</li>
                      <li>Detecting economic regimes that may affect model reliability</li>
                      <li>Providing business context for stakeholder communication</li>
                      <li>Explaining WHY predictions changed, not improving accuracy</li>
                    </ul>
                  </div>

                  <div className="bg-amber-50 rounded p-3 border border-amber-200">
                    <p className="font-medium text-amber-900 mb-1">
                      🔑 Key Insight: Why Not Use Economic Data for Prediction?
                    </p>
                    <p className="text-sm text-gray-700 mb-2">
                      In testing, adding macroeconomic features <strong>degraded accuracy</strong>:
                    </p>
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="bg-white rounded p-2 text-center">
                        <p className="font-semibold text-green-700">74 Features</p>
                        <p className="text-gray-600">0.26-2.22% MAPE</p>
                        <p className="text-green-600 font-medium">✅ Excellent</p>
                      </div>
                      <div className="bg-white rounded p-2 text-center">
                        <p className="font-semibold text-red-700">242 Features</p>
                        <p className="text-gray-600">7-12% MAPE</p>
                        <p className="text-red-600 font-medium">❌ Degraded</p>
                      </div>
                    </div>
                    <p className="text-xs text-gray-600 mt-2">
                      <strong>Why?</strong> Economic indicators move slowly and introduce overfitting.
                      Time-series features capture recent patterns more accurately.
                    </p>
                  </div>
                </div>
              </div>

              <div className="pt-2 border-t border-blue-200">
                <p className="text-xs text-blue-800">
                  <strong>Summary:</strong> RetailPRED achieves superior accuracy using only time-series
                  features from retail sales data. Economic context helps you understand and communicate
                  predictions, but doesn't drive them.
                </p>
              </div>
            </div>
          </AlertDescription>
        </Alert>
      )}
    </div>
  )
}
