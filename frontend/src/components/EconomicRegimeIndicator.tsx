/**
 * EconomicRegimeIndicator Component
 *
 * Displays economic regime warnings and model confidence indicators.
 *
 * IMPORTANT: Economic regime is CONTEXT ONLY - not used for model predictions.
 * Models use only 74 time-series features from MRTS data (0.26-2.22% MAPE).
 * This component helps stakeholders understand when to trust predictions.
 */

import React from 'react'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'

interface Trends {
  unemployment: 'stable' | 'rising' | 'falling'
  consumer_confidence: 'stable' | 'rising' | 'falling'
}

interface EconomicRegime {
  regime: 'normal' | 'expansion' | 'recession' | 'crisis'
  confidence: 'high' | 'medium' | 'low'
  trends?: Trends
  explanation: string
}

interface RegimeWithHistory extends EconomicRegime {
  date: string
  isUnusual: boolean
  modelReliability: 'high' | 'medium' | 'low'
}

interface EconomicRegimeIndicatorProps {
  regime: RegimeWithHistory
  showExplanation?: boolean
  compact?: boolean
}

export function EconomicRegimeIndicator({
  regime,
  showExplanation = true,
  compact = false
}: EconomicRegimeIndicatorProps) {
  // Get regime styling
  const getRegimeStyling = () => {
    switch (regime.regime) {
      case 'crisis':
        return {
          icon: '⚠️',
          color: 'red',
          bgColor: 'bg-red-50',
          borderColor: 'border-red-300',
          textColor: 'text-red-900',
          iconBg: 'bg-red-100',
          progressColor: 'bg-red-500'
        }
      case 'recession':
        return {
          icon: '📉',
          color: 'orange',
          bgColor: 'bg-orange-50',
          borderColor: 'border-orange-300',
          textColor: 'text-orange-900',
          iconBg: 'bg-orange-100',
          progressColor: 'bg-orange-500'
        }
      case 'expansion':
        return {
          icon: '📈',
          color: 'green',
          bgColor: 'bg-green-50',
          borderColor: 'border-green-300',
          textColor: 'text-green-900',
          iconBg: 'bg-green-100',
          progressColor: 'bg-green-500'
        }
      default:
        return {
          icon: '📊',
          color: 'blue',
          bgColor: 'bg-blue-50',
          borderColor: 'border-blue-300',
          textColor: 'text-blue-900',
          iconBg: 'bg-blue-100',
          progressColor: 'bg-blue-500'
        }
    }
  }

  const styling = getRegimeStyling()

  // Get reliability percentage
  const getReliabilityPercentage = () => {
    switch (regime.modelReliability) {
      case 'high':
        return 90
      case 'medium':
        return 60
      case 'low':
        return 30
      default:
        return 90
    }
  }

  // Get trend icon
  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'rising':
        return '↑'
      case 'falling':
        return '↓'
      default:
        return '→'
    }
  }

  // Get trend color
  const getTrendColor = (trend: string, indicator: string) => {
    if (indicator === 'unemployment') {
      // Rising unemployment is bad
      return trend === 'rising' ? 'text-red-600' :
             trend === 'falling' ? 'text-green-600' : 'text-gray-600'
    } else {
      // Rising confidence is good
      return trend === 'rising' ? 'text-green-600' :
             trend === 'falling' ? 'text-red-600' : 'text-gray-600'
    }
  }

  // Compact mode (small badge)
  if (compact) {
    return (
      <Badge
        variant="outline"
        className={cn(
          'text-xs font-medium',
          styling.bgColor,
          styling.borderColor,
          regime.isUnusual && 'border-2'
        )}
      >
        <span className="mr-1">{styling.icon}</span>
        {regime.regime.charAt(0).toUpperCase() + regime.regime.slice(1)}
      </Badge>
    )
  }

  // Full mode (alert component)
  return (
    <Alert
      className={cn(
        'mt-4',
        styling.bgColor,
        styling.borderColor,
        regime.isUnusual && 'border-2'
      )}
    >
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className={cn(
          'flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center text-xl',
          styling.iconBg
        )}>
          {styling.icon}
        </div>

        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-center gap-2 mb-2">
            <h4 className={cn('font-semibold', styling.textColor)}>
              Economic Regime: {regime.regime.charAt(0).toUpperCase() + regime.regime.slice(1)}
            </h4>
            {regime.isUnusual && (
              <Badge variant="outline" className="text-xs border-2">
                Unusual Conditions
              </Badge>
            )}
          </div>

          {/* Model Reliability Indicator */}
          <div className="mb-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-gray-600">
                Model Reliability
              </span>
              <span className={cn(
                'text-xs font-semibold',
                regime.modelReliability === 'high' ? 'text-green-700' :
                regime.modelReliability === 'medium' ? 'text-orange-700' :
                'text-red-700'
              )}>
                {regime.modelReliability.charAt(0).toUpperCase() + regime.modelReliability.slice(1)}
              </span>
            </div>
            <Progress
              value={getReliabilityPercentage()}
              className={cn('h-2', styling.bgColor)}
            />
            {regime.isUnusual && (
              <p className="text-xs text-gray-600 mt-1">
                ⚠️ Model was trained on normal economic conditions. Predictions during {regime.regime} periods may have higher uncertainty.
              </p>
            )}
          </div>

          {/* Economic Trends */}
          {regime.trends && (
            <div className={cn(
              'grid grid-cols-2 gap-2 p-2 rounded-lg mb-3',
              'bg-white bg-opacity-50'
            )}>
              <div className="text-sm">
                <div className="text-xs text-gray-600">Unemployment</div>
                <div className={cn(
                  'font-medium',
                  getTrendColor(regime.trends.unemployment, 'unemployment')
                )}>
                  {getTrendIcon(regime.trends.unemployment)}{' '}
                  {regime.trends.unemployment.charAt(0).toUpperCase() +
                   regime.trends.unemployment.slice(1)}
                </div>
              </div>
              <div className="text-sm">
                <div className="text-xs text-gray-600">Consumer Confidence</div>
                <div className={cn(
                  'font-medium',
                  getTrendColor(regime.trends.consumer_confidence, 'confidence')
                )}>
                  {getTrendIcon(regime.trends.consumer_confidence)}{' '}
                  {regime.trends.consumer_confidence.charAt(0).toUpperCase() +
                   regime.trends.consumer_confidence.slice(1)}
                </div>
              </div>
            </div>
          )}

          {/* Explanation */}
          {showExplanation && (
            <div className={cn(
              'p-2 rounded text-sm',
              'bg-white bg-opacity-50'
            )}>
              <p className="text-gray-700">
                <strong>Context:</strong> {regime.explanation}
              </p>
            </div>
          )}

          {/* Important Note */}
          <p className="text-xs text-gray-600 mt-2 italic">
            💡 This economic regime assessment is for interpretation only.
            The model uses only 74 time-series features from retail sales data
            (0.26-2.22% MAPE). Economic indicators are NOT used for predictions.
          </p>
        </div>
      </div>
    </Alert>
  )
}

/**
 * CompactRegimeBadge - Minimal version for inline display
 */
interface CompactRegimeBadgeProps {
  regime: RegimeWithHistory
  showLabel?: boolean
}

export function CompactRegimeBadge({
  regime,
  showLabel = true
}: CompactRegimeBadgeProps) {
  const styling = (() => {
    switch (regime.regime) {
      case 'crisis':
        return { icon: '⚠️', bgColor: 'bg-red-100', textColor: 'text-red-900' }
      case 'recession':
        return { icon: '📉', bgColor: 'bg-orange-100', textColor: 'text-orange-900' }
      case 'expansion':
        return { icon: '📈', bgColor: 'bg-green-100', textColor: 'text-green-900' }
      default:
        return { icon: '📊', bgColor: 'bg-blue-100', textColor: 'text-blue-900' }
    }
  })()

  return (
    <div className={cn(
      'inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium',
      styling.bgColor,
      styling.textColor
    )}>
      <span>{styling.icon}</span>
      {showLabel && (
        <span>
          {regime.regime.charAt(0).toUpperCase() + regime.regime.slice(1)}
        </span>
      )}
    </div>
  )
}
