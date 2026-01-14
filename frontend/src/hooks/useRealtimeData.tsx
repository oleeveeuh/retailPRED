/**
 * useRealtimeData Hook
 * Real-time data polling with automatic updates and notifications
 */

import React, { useEffect, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { RefreshCw, Wifi, WifiOff, AlertCircle } from 'lucide-react';

// Types
export interface RealtimeDataOptions<T> {
  queryKey: readonly unknown[];
  queryFn: () => Promise<T>;
  refetchInterval?: number; // in milliseconds, default 30000 (30s)
  enabled?: boolean;
  onSuccess?: (data: T) => void;
  onError?: (error: Error) => void;
  showToast?: boolean;
  toastMessage?: string;
}

export interface RealtimeDataResult<T> {
  data: T | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  isRefetching: boolean;
  lastUpdated: Date | null;
  refetch: () => void;
  networkStatus: 'online' | 'offline' | 'fetching';
}

// Network status hook
export const useNetworkStatus = () => {
  const [isOnline, setIsOnline] = useState(typeof navigator !== 'undefined' ? navigator.onLine : true);

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return isOnline;
};

// Main realtime data hook
export function useRealtimeData<T>({
  queryKey,
  queryFn,
  refetchInterval = 30000, // 30 seconds default
  enabled = true,
  onSuccess,
  onError,
  showToast = true,
  toastMessage = 'New data available',
}: RealtimeDataOptions<T>): RealtimeDataResult<T> {
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [isRefetching, setIsRefetching] = useState(false);
  const [networkStatus, setNetworkStatus] = useState<'online' | 'offline' | 'fetching'>('online');
  const queryClient = useQueryClient();
  const isOnline = useNetworkStatus();
  const previousDataRef = useRef<T | undefined>(undefined);

  // Set network status based on connectivity
  useEffect(() => {
    if (!isOnline) {
      setNetworkStatus('offline');
    } else {
      setNetworkStatus('online');
    }
  }, [isOnline]);

  // Main query with refetch interval
  const {
    data,
    isLoading,
    isError,
    error,
    refetch,
    isFetching,
  } = useQuery({
    queryKey,
    queryFn,
    refetchInterval: isOnline && enabled ? refetchInterval : false,
    enabled,
    onSuccess: (newData) => {
      setLastUpdated(new Date());

      // Check if data actually changed
      const hasChanged = JSON.stringify(previousDataRef.current) !== JSON.stringify(newData);

      if (hasChanged && previousDataRef.current !== undefined && showToast) {
        // Show toast for new data
        toast.success(toastMessage, {
          icon: <RefreshCw className="w-5 h-5 text-primary" />,
          duration: 4000,
          position: 'top-right',
        });
      }

      previousDataRef.current = newData;
      onSuccess?.(newData);
    },
    onError: (err) => {
      setNetworkStatus('offline');
      onError?.(err);
    },
  });

  // Update refetching status
  useEffect(() => {
    if (isFetching && !isLoading) {
      setIsRefetching(true);
      setNetworkStatus('fetching');
    } else {
      setIsRefetching(false);
      if (isOnline && networkStatus === 'fetching') {
        setNetworkStatus('online');
      }
    }
  }, [isFetching, isLoading, isOnline, networkStatus]);

  return {
    data,
    isLoading,
    isError,
    error,
    isRefetching,
    lastUpdated,
    refetch,
    networkStatus,
  };
};

// Hook for live prediction tracking
export interface LivePrediction {
  id: string;
  timestamp: Date;
  status: 'queued' | 'processing' | 'completed' | 'failed';
  category: string;
  model: string;
  progress?: number;
}

export const useLivePredictions = () => {
  const [predictions, setPredictions] = useState<LivePrediction[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const queryClient = useQueryClient();

  const addPrediction = (prediction: Omit<LivePrediction, 'id' | 'timestamp'>) => {
    const newPrediction: LivePrediction = {
      ...prediction,
      id: `pred-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date(),
    };

    setPredictions(prev => [newPrediction, ...prev]);
    setIsProcessing(true);

    // Simulate processing (in real app, this would be WebSocket/long-polling)
    setTimeout(() => {
      updatePrediction(newPrediction.id, { status: 'processing', progress: 25 });

      setTimeout(() => {
        updatePrediction(newPrediction.id, { status: 'processing', progress: 50 });

        setTimeout(() => {
          updatePrediction(newPrediction.id, { status: 'processing', progress: 75 });

          setTimeout(() => {
            updatePrediction(newPrediction.id, { status: 'completed', progress: 100 });
            setIsProcessing(false);

            // Refresh data after prediction completes
            queryClient.invalidateQueries({ queryKey: ['predictions'] });

            toast.success('Prediction generated successfully!', {
              icon: <RefreshCw className="w-5 h-5 text-emerald-500" />,
              duration: 3000,
            });
          }, 500);
        }, 500);
      }, 500);
    }, 500);

    return newPrediction.id;
  };

  const updatePrediction = (id: string, updates: Partial<LivePrediction>) => {
    setPredictions(prev =>
      prev.map(p => (p.id === id ? { ...p, ...updates } : p))
    );
  };

  const removePrediction = (id: string) => {
    setPredictions(prev => prev.filter(p => p.id !== id));
  };

  const clearCompleted = () => {
    setPredictions(prev => prev.filter(p => p.status !== 'completed'));
  };

  return {
    predictions,
    isProcessing,
    addPrediction,
    updatePrediction,
    removePrediction,
    clearCompleted,
  };
};

// Network status indicator component
export const NetworkStatusIndicator = () => {
  const [status, setStatus] = useState<'online' | 'offline'>('online');

  useEffect(() => {
    const handleOnline = () => setStatus('online');
    const handleOffline = () => setStatus('offline');

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    setStatus(navigator.onLine ? 'online' : 'offline');

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return (
    <div
      className={`fixed bottom-4 right-4 z-50 flex items-center gap-2 px-4 py-2 rounded-full shadow-lg backdrop-blur-xl ${
        status === 'online'
          ? 'bg-emerald-100/90 dark:bg-emerald-900/90 text-emerald-700 dark:text-emerald-300'
          : 'bg-red-100/90 dark:bg-red-900/90 text-red-700 dark:text-red-300'
      }`}
    >
      {status === 'online' ? (
        <>
          <Wifi className="w-4 h-4 animate-pulse" />
          <span className="text-sm font-medium">Online</span>
        </>
      ) : (
        <>
          <WifiOff className="w-4 h-4" />
          <span className="text-sm font-medium">Offline</span>
        </>
      )}
    </div>
  );
};

// Last updated timestamp display
export const LastUpdatedDisplay = ({ lastUpdated }: { lastUpdated: Date | null }) => {
  const [timeAgo, setTimeAgo] = useState('');

  useEffect(() => {
    if (!lastUpdated) {
      setTimeAgo('Never');
      return;
    }

    const updateTimeAgo = () => {
      const now = new Date();
      const diff = now.getTime() - lastUpdated.getTime();
      const seconds = Math.floor(diff / 1000);
      const minutes = Math.floor(seconds / 60);
      const hours = Math.floor(minutes / 60);

      if (seconds < 60) {
        setTimeAgo(`${seconds}s ago`);
      } else if (minutes < 60) {
        setTimeAgo(`${minutes}m ago`);
      } else if (hours < 24) {
        setTimeAgo(`${hours}h ago`);
      } else {
        setTimeAgo(lastUpdated.toLocaleDateString());
      }
    };

    updateTimeAgo();
    const interval = setInterval(updateTimeAgo, 1000);

    return () => clearInterval(interval);
  }, [lastUpdated]);

  return (
    <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400">
      <RefreshCw className="w-4 h-4" />
      <span>Updated {timeAgo}</span>
    </div>
  );
};

// Real-time training progress hook
export const useTrainingProgress = () => {
  const [trainingState, setTrainingState] = useState<{
    isTraining: boolean;
    progress: number;
    stage: string;
    model: string;
  }>({
    isTraining: false,
    progress: 0,
    stage: '',
    model: '',
  });

  const startTraining = (model: string) => {
    setTrainingState({
      isTraining: true,
      progress: 0,
      stage: 'Initializing...',
      model,
    });

    // Simulate training progress
    const stages = [
      { progress: 10, stage: 'Loading data...' },
      { progress: 25, stage: 'Preprocessing features...' },
      { progress: 40, stage: 'Training model...' },
      { progress: 60, stage: 'Validating...' },
      { progress: 80, stage: 'Saving model...' },
      { progress: 95, stage: 'Finalizing...' },
      { progress: 100, stage: 'Complete!' },
    ];

    stages.forEach(({ progress, stage }) => {
      setTimeout(() => {
        setTrainingState(prev => ({
          ...prev,
          progress,
          stage,
        }));
      }, progress * 100); // Simulate time based on progress
    });

    setTimeout(() => {
      setTrainingState({
        isTraining: false,
        progress: 100,
        stage: 'Complete!',
        model,
      });

      toast.success(`Model "${model}" training complete!`, {
        icon: <RefreshCw className="w-5 h-5 text-emerald-500" />,
        duration: 5000,
      });
    }, 10000); // 10 seconds total
  };

  return {
    ...trainingState,
    startTraining,
  };
};
