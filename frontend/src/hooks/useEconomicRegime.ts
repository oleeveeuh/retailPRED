/**
 * useEconomicRegime Hook
 *
 * Fetches economic regime classification for model confidence assessment.
 *
 * IMPORTANT: This is for INTERPRETATION ONLY - not used for model predictions.
 * Models use only 74 time-series features from MRTS data (0.26-2.22% MAPE).
 */

import { useState, useEffect } from 'react'
import { api } from '@/api/client'

export interface EconomicRegime {
  regime: 'normal' | 'expansion' | 'recession' | 'crisis'
  confidence: 'high' | 'medium' | 'low'
  trends?: {
    unemployment: 'stable' | 'rising' | 'falling'
    consumer_confidence: 'stable' | 'rising' | 'falling'
  }
  explanation: string
}

export interface RegimeWithHistory extends EconomicRegime {
  date: string
  isUnusual: boolean
  modelReliability: 'high' | 'medium' | 'low'
}

/**
 * Hook to fetch economic regime for a specific date
 *
 * @param date - Date string in format 'YYYY-MM-DD'
 * @returns Economic regime classification with model reliability assessment
 */
export function useEconomicRegime(date: string | null) {
  const [regime, setRegime] = useState<RegimeWithHistory | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (!date) {
      setRegime(null)
      return
    }

    setLoading(true)
    setError(null)

    // In demo mode, load from static file
    if (import.meta.env.VITE_DEMO_MODE === 'true') {
      import('@/services/demoDataService').then(({ demoDataService }) => {
        demoDataService.getEconomicRegime(date).then((data: EconomicRegime) => {
          const regimeWithHistory: RegimeWithHistory = {
            ...data,
            date,
            isUnusual: data.regime !== 'normal',
            modelReliability: getModelReliability(data.regime)
          }
          setRegime(regimeWithHistory)
        }).catch(setError)
      }).finally(() => setLoading(false))
    } else {
      // In production, fetch from API
      api.get(`/api/context/regime/${date}`)
        .then((data: EconomicRegime) => {
          const regimeWithHistory: RegimeWithHistory = {
            ...data,
            date,
            isUnusual: data.regime !== 'normal',
            modelReliability: getModelReliability(data.regime)
          }
          setRegime(regimeWithHistory)
        })
        .catch(setError)
        .finally(() => setLoading(false))
    }
  }, [date])

  return { regime, loading, error }
}

/**
 * Hook to get current economic regime (most recent)
 *
 * @returns Current economic regime
 */
export function useCurrentRegime() {
  const [regime, setRegime] = useState<RegimeWithHistory | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)

    // In demo mode, load from static file
    if (import.meta.env.VITE_DEMO_MODE === 'true') {
      import('@/services/demoDataService').then(({ demoDataService }) => {
        demoDataService.getCurrentRegime().then((data: EconomicRegime) => {
          const date = new Date().toISOString().split('T')[0]
          const regimeWithHistory: RegimeWithHistory = {
            ...data,
            date,
            isUnusual: data.regime !== 'normal',
            modelReliability: getModelReliability(data.regime)
          }
          setRegime(regimeWithHistory)
        }).catch(setError)
      }).finally(() => setLoading(false))
    } else {
      // In production, fetch from API
      api.get('/api/context/summary')
        .then((data: any) => {
          const date = new Date().toISOString().split('T')[0]
          const regimeWithHistory: RegimeWithHistory = {
            ...data.regime,
            date,
            isUnusual: data.regime.regime !== 'normal',
            modelReliability: getModelReliability(data.regime.regime)
          }
          setRegime(regimeWithHistory)
        })
        .catch(setError)
        .finally(() => setLoading(false))
    }
  }, [])

  return { regime, loading, error }
}

/**
 * Hook to get regime history
 *
 * @param startDate - Optional start date
 * @param endDate - Optional end date
 * @returns Array of historical regime classifications
 */
export function useRegimeHistory(startDate?: string, endDate?: string) {
  const [regimes, setRegimes] = useState<RegimeWithHistory[]>([])
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
        demoDataService.getRegimeHistory(startDate, endDate)
          .then((data: EconomicRegime[]) => {
            const regimesWithHistory = data.map((r, idx) => ({
              ...r,
              date: r.date || new Date().toISOString().split('T')[0],
              isUnusual: r.regime !== 'normal',
              modelReliability: getModelReliability(r.regime)
            }))
            setRegimes(regimesWithHistory)
          })
          .catch(setError)
      }).finally(() => setLoading(false))
    } else {
      // In production, fetch from API
      api.get(`/api/context/regime/history?${params}`)
        .then((data: EconomicRegime[]) => {
          const regimesWithHistory = data.map(r => ({
            ...r,
            date: r.date || new Date().toISOString().split('T')[0],
            isUnusual: r.regime !== 'normal',
            modelReliability: getModelReliability(r.regime)
          }))
          setRegimes(regimesWithHistory)
        })
        .catch(setError)
        .finally(() => setLoading(false))
    }
  }, [startDate, endDate])

  return { regimes, loading, error }
}

/**
 * Helper function to determine model reliability based on regime
 *
 * @param regime - Economic regime classification
 * @returns Model reliability level
 */
function getModelReliability(regime: string): 'high' | 'medium' | 'low' {
  switch (regime) {
    case 'crisis':
      return 'low'
    case 'recession':
      return 'medium'
    case 'expansion':
      return 'high'
    default:
      return 'high'
  }
}
