/**
 * useAnomalyDetection Hook
 *
 * Detects anomalous predictions based on percentage changes.
 * Anomalies are defined as predictions with >5% change from previous.
 */

import { useMemo } from 'react'

export interface Prediction {
  date: string
  value: number
  predicted_value?: number
  category?: string
  model_type?: string
}

export interface AnomalyPrediction extends Prediction {
  isAnomaly: boolean
  change: number
  severity: 'moderate' | 'severe'
  changeMagnitude: number
}

/**
 * Hook to detect anomalies in prediction time series
 *
 * @param predictions - Array of predictions in chronological order
 * @param threshold - Percentage change threshold (default: 5%)
 * @returns Predictions with anomaly flags and calculated changes
 */
export function useAnomalyDetection(
  predictions: Prediction[],
  threshold: number = 5
): AnomalyPrediction[] {
  return useMemo(() => {
    return predictions.map((pred, idx) => {
      if (idx === 0) {
        return {
          ...pred,
          isAnomaly: false,
          change: 0,
          severity: 'moderate',
          changeMagnitude: 0
        }
      }

      const prevPred = predictions[idx - 1]
      const prevValue = prevPred.value || prevPred.predicted_value || pred.value

      const change = ((pred.value - prevValue) / prevValue) * 100
      const changeMagnitude = Math.abs(change)
      const isAnomaly = changeMagnitude > threshold
      const severity: 'moderate' | 'severe' = changeMagnitude > 10 ? 'severe' : 'moderate'

      return {
        ...pred,
        isAnomaly,
        change,
        severity,
        changeMagnitude
      }
    })
  }, [predictions, threshold])
}

/**
 * Hook to get only anomalous predictions
 *
 * @param predictions - Array of predictions
 * @returns Filtered array of only anomalous predictions
 */
export function useAnomaliesOnly(
  predictions: Prediction[],
  threshold: number = 5
): AnomalyPrediction[] {
  const predictionsWithAnomalies = useAnomalyDetection(predictions, threshold)

  return useMemo(() => {
    return predictionsWithAnomalies.filter(p => p.isAnomaly)
  }, [predictionsWithAnomalies])
}

/**
 * Hook to get anomaly statistics
 *
 * @param predictions - Array of predictions
 * @returns Statistics about anomalies
 */
export function useAnomalyStats(predictions: Prediction[]) {
  const predictionsWithAnomalies = useAnomalyDetection(predictions)

  return useMemo(() => {
    const anomalies = predictionsWithAnomalies.filter(p => p.isAnomaly)

    const severeAnomalies = anomalies.filter(p => p.severity === 'severe')
    const moderateAnomalies = anomalies.filter(p => p.severity === 'moderate')

    const positiveAnomalies = anomalies.filter(p => p.change > 0)
    const negativeAnomalies = anomalies.filter(p => p.change < 0)

    const avgChange = anomalies.length > 0
      ? anomalies.reduce((sum, p) => sum + p.changeMagnitude, 0) / anomalies.length
      : 0

    const maxChange = anomalies.length > 0
      ? Math.max(...anomalies.map(p => p.changeMagnitude))
      : 0

    return {
      total: anomalies.length,
      severe: severeAnomalies.length,
      moderate: moderateAnomalies.length,
      positive: positiveAnomalies.length,
      negative: negativeAnomalies.length,
      averageChange: avgChange,
      maxChange
    }
  }, [predictionsWithAnomalies])
}
