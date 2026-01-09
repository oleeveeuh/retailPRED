/**
 * EconomicContextSettings Component
 *
 * Settings panel for economic context display preferences.
 */

import React, { useEffect } from 'react'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Card } from '@/components/ui/card'

interface EconomicContextSettingsProps {
  showEconomicContext?: boolean
  onToggle?: (show: boolean) => void
}

export function EconomicContextSettings({
  showEconomicContext: externalShow,
  onToggle: externalOnToggle
}: EconomicContextSettingsProps) {
  // Internal state (component-controlled)
  const [internalShow, setInternalShow] = React.useState(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('show-economic-context')
      return stored ? JSON.parse(stored) : true
    }
    return true
  })

  // Determine if controlled or uncontrolled
  const isControlled = externalShow !== undefined
  const showEconomicContext = isControlled ? externalShow : internalShow

  const handleToggle = (checked: boolean) => {
    if (isControlled && externalOnToggle) {
      externalOnToggle(checked)
    } else {
      setInternalShow(checked)
      localStorage.setItem('show-economic-context', JSON.stringify(checked))
    }
  }

  return (
    <Card className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 border-blue-200">
      <div className="space-y-4">
        <div>
          <h3 className="font-semibold text-gray-900 mb-1">
            Economic Context Display
          </h3>
          <p className="text-sm text-gray-600">
            Configure how economic indicators are shown in the application
          </p>
        </div>

        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex-1">
              <Label htmlFor="show-economic-context" className="text-gray-900">
                Show Economic Context
              </Label>
              <p className="text-xs text-gray-600 mt-1">
                Display economic indicators for interpretation (not used in predictions)
              </p>
            </div>
            <Switch
              id="show-economic-context"
              checked={showEconomicContext}
              onCheckedChange={handleToggle}
            />
          </div>

          {showEconomicContext && (
            <div className="ml-4 space-y-2 text-sm text-gray-600 bg-white rounded p-3">
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span>Economic regime indicator</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span>Anomaly explanations</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span>Historical event annotations</span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-green-600">✓</span>
                <span>Event timeline</span>
              </div>
            </div>
          )}
        </div>

        <div className="pt-3 border-t border-blue-200">
          <p className="text-xs text-gray-500">
            💡 <strong>Note:</strong> Economic context is for interpretation only.
            The model uses 74 time-series features from retail sales data (0.26% MAPE).
          </p>
        </div>
      </div>
    </Card>
  )
}

// Hook for using economic context preference
export function useEconomicContextPreference(): [boolean, (show: boolean) => void] {
  const [showEconomicContext, setShowEconomicContext] = React.useState(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('show-economic-context')
      return stored ? JSON.parse(stored) : true
    }
    return true
  })

  const toggleEconomicContext = (show: boolean) => {
    setShowEconomicContext(show)
    localStorage.setItem('show-economic-context', JSON.stringify(show))
  }

  return [showEconomicContext, toggleEconomicContext]
}
