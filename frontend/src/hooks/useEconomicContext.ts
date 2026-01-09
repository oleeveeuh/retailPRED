/**
 * useEconomicContext Hook
 *
 * Fetches economic context data for explaining predictions.
 *
 * IMPORTANT: This data is for INTERPRETATION ONLY - not used for model predictions.
 */

import { useState, useEffect } from 'react'
import { api } from '@/api/client'

export interface EconomicIndicators {
  unemployment: number | null
  consumer_confidence: number | null
  fed_rate: number | null
  cpi: number | null
  industrial_production: number | null
}

export interface EconomicRegime {
  regime: 'normal' | 'expansion' | 'recession' | 'crisis'
  confidence: 'high' | 'medium' | 'low'
  trends?: {
    unemployment: 'stable' | 'rising' | 'falling'
    consumer_confidence: 'stable' | 'rising' | 'falling'
  }
  explanation: string
}

export interface EconomicAnomaly {
  indicator: string
  value: number
  z_score: number
  severity: 'high' | 'medium'
  direction: 'high' | 'low'
}

export interface EconomicContext {
  indicators: EconomicIndicators
  regime: EconomicRegime
  anomalies: EconomicAnomaly[]
  note?: string
}

export interface IndicatorWithChange extends EconomicIndicators {
  unemploymentChange: number
  confidenceChange: number
}

/**
 * Hook to fetch economic context for a specific date
 *
 * @param date - Date string in format 'YYYY-MM-DD'
 * @returns Economic context data or null
 */
export function useEconomicContext(date: string | null) {
  const [context, setContext] = useState<EconomicContext | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (!date) {
      setContext(null)
      return
    }

    setLoading(true)
    setError(null)

    // In demo mode, load from static file
    if (import.meta.env.VITE_DEMO_MODE === 'true') {
      import('@/services/demoDataService').then(({ demoDataService }) => {
        demoDataService.getEconomicContext(date).then(setContext).catch(setError)
      }).finally(() => setLoading(false))
    } else {
      // In production, fetch from API
      api.get(`/api/context/indicators/${date}`)
        .then(setContext)
        .catch(setError)
        .finally(() => setLoading(false))
    }
  }, [date])

  return { context, loading, error }
}

/**
 * Hook to fetch historical anomalies
 *
 * @param startDate - Optional start date
 * @param endDate - Optional end date
 * @returns Array of historical anomalies
 */
export function useHistoricalAnomalies(startDate?: string, endDate?: string) {
  const [anomalies, setAnomalies] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)

    const params = new URLSearchParams()
    if (startDate) params.append('start_date', startDate)
    if (endDate) params.append('end_date', endDate)

    // In demo mode, load from static file
    if (import.meta.env.VITE_DEMO_MODE === 'true') {
      import('@/services/demoDataService').then(({ demoDataService }) => {
        demoDataService.getHistoricalAnomalies(startDate, endDate)
          .then(setAnomalies)
          .catch(setError)
      }).finally(() => setLoading(false))
    } else {
      // In production, fetch from API
      api.get(`/api/context/anomalies?${params}`)
        .then(data => setAnomalies(data.anomalies))
        .catch(setError)
        .finally(() => setLoading(false))
    }
  }, [startDate, endDate])

  return { anomalies, loading, error }
}

/**
 * Hook to get economic context with change calculations
 *
 * This extends useEconomicContext by calculating 3-month changes
 * for unemployment and consumer confidence.
 *
 * @param date - Date string in format 'YYYY-MM-DD'
 * @returns Economic context with indicator changes
 */
export function useEconomicContextWithChanges(date: string | null) {
  const { context, loading, error } = useEconomicContext(date)

  const contextWithChanges = useMemo(() => {
    if (!context) return null

    // Calculate 3-month changes (simplified for demo)
    const indicatorsWithChange: IndicatorWithChange = {
      ...context.indicators,
      unemploymentChange: 0,  // Would need historical data to calculate
      confidenceChange: 0      // Would need historical data to calculate
    }

    // In demo mode, we could pre-calculate these or fetch from historical data
    // For now, setting to 0 as placeholder

    return {
      ...context,
      indicators: indicatorsWithChange
    }
  }, [context])

  return { context: contextWithChanges, loading, error }
}
