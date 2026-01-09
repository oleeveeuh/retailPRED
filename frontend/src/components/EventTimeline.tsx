/**
 * EventTimeline Component
 *
 * Displays a timeline of major economic events for context.
 *
 * IMPORTANT: These events are for INTERPRETATION ONLY - not used for model predictions.
 * Models use only 74 time-series features from MRTS data (0.26-2.22% MAPE).
 */

import React from 'react'
import { HistoricalEvent } from './ForecastChart'

interface EventTimelineProps {
  events: HistoricalEvent[]
  title?: string
  showContext?: boolean
}

export function EventTimeline({
  events,
  title = 'Historical Economic Events',
  showContext = true
}: EventTimelineProps) {
  // Sort events by date (most recent first)
  const sortedEvents = [...events].sort(
    (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
  )

  // Get event icon and color
  const getEventStyling = (type: string) => {
    switch (type) {
      case 'crisis':
        return {
          icon: '🚨',
          bgColor: 'bg-red-50',
          borderColor: 'border-red-200',
          textColor: 'text-red-900'
        }
      case 'recession':
        return {
          icon: '⚠️',
          bgColor: 'bg-orange-50',
          borderColor: 'border-orange-200',
          textColor: 'text-orange-900'
        }
      case 'expansion':
        return {
          icon: '📈',
          bgColor: 'bg-green-50',
          borderColor: 'border-green-200',
          textColor: 'text-green-900'
        }
      default:
        return {
          icon: '📊',
          bgColor: 'bg-blue-50',
          borderColor: 'border-blue-200',
          textColor: 'text-blue-900'
        }
    }
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <div className="mb-4">
        <h3 className="font-semibold text-lg text-gray-900">{title}</h3>
        {showContext && (
          <p className="text-sm text-gray-500 mt-1">
            Context for understanding prediction anomalies
          </p>
        )}
      </div>

      {/* Important note */}
      {showContext && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded">
          <p className="text-xs text-blue-900">
            <strong>💡 Important:</strong> These events provide economic context for interpreting
            historical predictions. Models use only 74 time-series features from retail sales data
            (0.26-2.22% MAPE). Economic indicators are NOT used for predictions.
          </p>
        </div>
      )}

      {/* Timeline */}
      <div className="space-y-4">
        {sortedEvents.map((event, idx) => {
          const styling = getEventStyling(event.type)

          return (
            <div
              key={event.date}
              className={cn(
                'flex gap-4 p-4 rounded-lg border-2 transition-colors hover:shadow-md',
                styling.bgColor,
                styling.borderColor
              )}
            >
              {/* Icon */}
              <div className="flex-shrink-0 text-3xl">
                {styling.icon}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                {/* Header */}
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <h4 className={cn('font-semibold text-lg', styling.textColor)}>
                      {event.label}
                    </h4>
                    <p className="text-sm text-gray-600">
                      {new Date(event.date).toLocaleDateString('en-US', {
                        month: 'long',
                        year: 'numeric'
                      })}
                    </p>
                  </div>
                  <span className={cn(
                    'px-2 py-1 text-xs font-semibold rounded',
                    styling.bgColor,
                    styling.textColor,
                    'border border-current'
                  )}>
                    {event.type.charAt(0).toUpperCase() + event.type.slice(1)}
                  </span>
                </div>

                {/* Explanation */}
                <p className="text-sm text-gray-700 mb-3">
                  {event.explanation}
                </p>

                {/* Economic Context */}
                <div className="bg-white bg-opacity-60 rounded p-3">
                  <div className="text-xs text-gray-500 mb-2 font-medium">
                    Economic Context (Interpretation Only)
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    {event.economicContext.unemployment && (
                      <div>
                        <span className="text-gray-600">Unemployment:</span>{' '}
                        <span className="font-medium text-gray-900">
                          {event.economicContext.unemployment}%
                        </span>
                      </div>
                    )}
                    {event.economicContext.confidence && (
                      <div>
                        <span className="text-gray-600">Consumer Confidence:</span>{' '}
                        <span className="font-medium text-gray-900">
                          {event.economicContext.confidence}
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Timeline connector (except for last item) */}
              {idx < sortedEvents.length - 1 && (
                <div className="absolute left-6 mt-4 w-0.5 h-4 bg-gray-300" />
              )}
            </div>
          )
        })}
      </div>

      {/* Footer note */}
      {showContext && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <p className="text-xs text-gray-500">
            Events are sourced from FRED economic data. Click on event annotations in the
            forecast chart to see detailed context.
          </p>
        </div>
      )}
    </div>
  )
}

/**
 * CompactEventTimeline - Minimal version for smaller spaces
 */
interface CompactEventTimelineProps {
  events: HistoricalEvent[]
  maxEvents?: number
}

export function CompactEventTimeline({
  events,
  maxEvents = 5
}: CompactEventTimelineProps) {
  const sortedEvents = [...events]
    .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
    .slice(0, maxEvents)

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <h3 className="font-semibold text-sm text-gray-900 mb-3">
        Recent Economic Events
      </h3>
      <div className="space-y-2">
        {sortedEvents.map((event) => {
          const icon = event.type === 'crisis' ? '🚨' : '⚠️'
          const colorClass = event.type === 'crisis'
            ? 'text-red-700'
            : 'text-orange-700'

          return (
            <div
              key={event.date}
              className="flex items-start gap-2 text-sm p-2 rounded hover:bg-gray-50"
            >
              <span className="text-lg">{icon}</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span className={cn('font-medium', colorClass)}>
                    {event.label}
                  </span>
                  <span className="text-xs text-gray-500">
                    {new Date(event.date).toLocaleDateString('en-US', {
                      month: 'short',
                      year: '2-digit'
                    })}
                  </span>
                </div>
                <p className="text-xs text-gray-600 line-clamp-1">
                  {event.explanation}
                </p>
              </div>
            </div>
          )
        })}
      </div>
      <p className="text-xs text-gray-500 mt-3 italic">
        Context only - not used for predictions
      </p>
    </div>
  )
}

// Helper function for className conditionals
function cn(...classes: (string | boolean | undefined | null)[]) {
  return classes.filter(Boolean).join(' ')
}
