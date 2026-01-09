/**
 * AnomalyExplanation Component
 *
 * Displays economic context explanations for unusual predictions.
 *
 * IMPORTANT: Economic data is CONTEXT ONLY - not used for model predictions.
 * Models use only 74 time-series features from MRTS data (0.26-2.22% MAPE).
 */

import React from 'react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'

interface Indicator {
  unemployment: number
  unemploymentChange: number  // vs. 3mo prior
  consumerConfidence: number
  confidenceChange: number
  fedRate: number
}

interface EconomicContext {
  regime: 'normal' | 'expansion' | 'recession' | 'crisis'
  indicators: Indicator
  anomalies: string[]
  explanation: string
}

interface AnomalyExplanationProps {
  date: string
  predictionChange: number  // % change from previous
  economicContext?: EconomicContext
}

export function AnomalyExplanation({
  date,
  predictionChange,
  economicContext
}: AnomalyExplanationProps) {
  // Only show if there's a significant prediction change (>5%)
  if (Math.abs(predictionChange) < 5 || !economicContext) {
    return null
  }

  const isNegative = predictionChange < 0
  const severity = Math.abs(predictionChange) > 10 ? 'severe' : 'moderate'

  // Get regime icon and color
  const getRegimeIcon = () => {
    switch (economicContext.regime) {
      case 'crisis':
        return '⚠️'
      case 'recession':
        return '📉'
      case 'expansion':
        return '📈'
      default:
        return '📊'
    }
  }

  const getRegimeColor = () => {
    switch (economicContext.regime) {
      case 'crisis':
        return 'bg-red-50 border-red-200'
      case 'recession':
        return 'bg-orange-50 border-orange-200'
      case 'expansion':
        return 'bg-green-50 border-green-200'
      default:
        return 'bg-blue-50 border-blue-200'
    }
  }

  return (
    <Alert
      variant={isNegative ? 'destructive' : 'default'}
      className={cn(
        'mt-4',
        isNegative && 'border-red-300 bg-red-50'
      )}
    >
      <div className="flex items-start gap-3">
        <span className="text-2xl mt-0.5">{getRegimeIcon()}</span>

        <div className="flex-1">
          <h4 className="font-semibold mb-2">
            {isNegative ? 'Sales Decline' : 'Sales Surge'} Detected
          </h4>

          <p className="text-sm mb-3">
            <strong>{date}:</strong> Sales{' '}
            <span className={cn(
              'font-medium',
              isNegative ? 'text-red-700' : 'text-green-700'
            )}>
              {isNegative ? 'dropped' : 'increased'} by {Math.abs(predictionChange).toFixed(1)}%
            </span>
          </p>

          {/* Economic Context Panel */}
          <div className={cn(
            'border rounded-lg p-3 mb-3',
            getRegimeColor()
          )}>
            <div className="flex items-center gap-2 mb-2">
              <Badge variant="outline" className="font-semibold">
                {economicContext.regime.charAt(0).toUpperCase() +
                 economicContext.regime.slice(1)}
              </Badge>
              <span className="text-xs text-gray-600">
                Economic Context
              </span>
              <Badge variant="secondary" className="text-xs">
                Interpretation Only
              </Badge>
            </div>

            <div className="grid grid-cols-3 gap-3 text-sm">
              <IndicatorBadge
                label="Unemployment"
                value={`${economicContext.indicators.unemployment.toFixed(1)}%`}
                change={economicContext.indicators.unemploymentChange}
              />
              <IndicatorBadge
                label="Consumer Confidence"
                value={economicContext.indicators.consumerConfidence.toFixed(1)}
                change={economicContext.indicators.confidenceChange}
              />
              <IndicatorBadge
                label="Fed Rate"
                value={`${economicContext.indicators.fedRate.toFixed(2)}%`}
                change={undefined}
              />
            </div>

            {economicContext.anomalies.length > 0 && (
              <div className="mt-2 pt-2 border-t border-gray-300">
                <div className="text-xs text-gray-600 mb-1">
                  Anomalous Indicators:
                </div>
                <div className="flex flex-wrap gap-1">
                  {economicContext.anomalies.map((anomaly) => (
                    <Badge key={anomaly} variant="outline" className="text-xs">
                      {anomaly}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Explanation */}
          <div className="bg-white bg-opacity-50 rounded p-2 mb-2">
            <p className="text-sm">
              <strong>Interpretation:</strong> {economicContext.explanation}
            </p>
          </div>

          {/* Important Note */}
          <p className="text-xs text-gray-600 mt-2 italic">
            💡 The model predicted this change from recent sales patterns using 74
            time-series features. Economic indicators shown above provide
            context for interpretation but were <strong>NOT used</strong> in the
            prediction.
          </p>
        </div>
      </div>
    </Alert>
  )
}

// Helper component for individual indicators
function IndicatorBadge({
  label,
  value,
  change
}: {
  label: string
  value: string
  change?: number
}) {
  const isNegative = change !== undefined && change < 0
  const isSignificant = change !== undefined && Math.abs(change) > 1

  return (
    <div>
      <div className="text-xs text-gray-600">{label}</div>
      <div className="font-medium text-sm">{value}</div>
      {change !== undefined && (
        <div className={cn(
          "text-xs flex items-center gap-1",
          isSignificant && (isNegative ? "text-red-600" : "text-green-600")
        )}>
          {change > 0 ? '↑' : '↓'} {Math.abs(change).toFixed(1)}%
          {isSignificant && ' (!)'}
        </div>
      )}
    </div>
  )
}
