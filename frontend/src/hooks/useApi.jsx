/**
 * Custom hook for API calls with enhanced functionality
 * 
 * This hook provides:
 * - Loading, error, and data states for API requests
 * - Cache management for expensive requests
 * - Pagination helpers for list endpoints
 * - Polling for long-running operations
 * - Status tracking across components
 */
import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Cache implementation for API responses
 */
const apiCache = {
  cache: new Map(),
  
  // Get item from cache if not expired
  get: (key) => {
    const item = apiCache.cache.get(key);
    if (!item) return null;
    
    // Check if item has expired
    if (item.expiry && item.expiry < Date.now()) {
      apiCache.cache.delete(key);
      return null;
    }
    
    return item.data;
  },
  
  // Store item in cache with optional TTL in milliseconds
  set: (key, data, ttl = null) => {
    const item = {
      data,
      expiry: ttl ? Date.now() + ttl : null,
    };
    apiCache.cache.set(key, item);
  },
  
  // Invalidate specific cache keys matching a pattern
  invalidate: (pattern) => {
    for (const key of apiCache.cache.keys()) {
      if (key.startsWith(pattern)) {
        apiCache.cache.delete(key);
      }
    }
  },
  
  // Clear the entire cache
  clear: () => {
    apiCache.cache.clear();
  }
};

/**
 * Main API hook for handling data fetching
 */
export function useApi(apiFn, options = {}) {
  const {
    initialData = null,
    deps = [],
    cacheKey = null,
    cacheTTL = 5 * 60 * 1000, // 5 minutes by default
    enableCache = false,
    onSuccess = null,
    onError = null,
    pollingInterval = null,
    autoExecute = false,
  } = options;

  const [data, setData] = useState(initialData);
  const [loading, setLoading] = useState(autoExecute);
  const [error, setError] = useState(null);
  const [execCount, setExecCount] = useState(0);
  
  // Store the latest API function and options in a ref
  const apiRef = useRef({ apiFn, options });
  apiRef.current = { apiFn, options };
  
  const pollingTimerRef = useRef(null);
  
  // The main execute function that runs the API call
  const execute = useCallback(async (...args) => {
    // Don't use caching for mutations (non-GET requests)
    const isMutation = !enableCache;
    const currentCacheKey = isMutation ? null : (cacheKey || `${apiFn.name}:${JSON.stringify(args)}`);
    
    // Try to get from cache first if this is a GET request with caching enabled
    if (enableCache && currentCacheKey) {
      const cachedData = apiCache.get(currentCacheKey);
      if (cachedData) {
        setData(cachedData);
        return cachedData;
      }
    }
    
    setLoading(true);
    setError(null);
    
    try {
      const result = await apiFn(...args);
      setData(result);
      
      // Cache the result if needed
      if (enableCache && currentCacheKey && result) {
        apiCache.set(currentCacheKey, result, cacheTTL);
      }
      
      // Call success callback if provided
      if (onSuccess) {
        onSuccess(result);
      }
      
      return result;
    } catch (err) {
      setError(err);
      
      // Call error callback if provided
      if (onError) {
        onError(err);
      }
      
      throw err;
    } finally {
      setLoading(false);
      setExecCount(count => count + 1);
    }
  }, [apiFn, cacheKey, cacheTTL, enableCache, onSuccess, onError]);
  
  // Set up polling if enabled
  useEffect(() => {
    // Clear any existing polling timer
    if (pollingTimerRef.current) {
      clearInterval(pollingTimerRef.current);
      pollingTimerRef.current = null;
    }
    
    // Set up new polling if requested
    if (pollingInterval && autoExecute) {
      pollingTimerRef.current = setInterval(() => {
        const { apiFn, options } = apiRef.current;
        execute();
      }, pollingInterval);
    }
    
    return () => {
      if (pollingTimerRef.current) {
        clearInterval(pollingTimerRef.current);
      }
    };
  }, [pollingInterval, autoExecute, execute]);
  
  // Auto-execute on mount or when dependencies change
  useEffect(() => {
    if (autoExecute) {
      execute();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoExecute, execute, ...deps]);
  
  // Reset state when API function changes
  useEffect(() => {
    setData(initialData);
    setLoading(autoExecute);
    setError(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiFn]);
  
  // Helper for pagination handling
  const paginated = {
    page: data?.page || 1,
    totalPages: data?.totalPages || 1,
    pageSize: data?.pageSize || 10,
    total: data?.total || 0,
    hasNext: data?.page < data?.totalPages,
    hasPrevious: data?.page > 1,
    goToPage: (page) => execute(page, data?.pageSize || 10),
    nextPage: () => execute(data?.page + 1 || 2, data?.pageSize || 10),
    previousPage: () => execute(Math.max(1, (data?.page || 2) - 1), data?.pageSize || 10),
  };
  
  // Helper for manual cache control
  const cache = {
    invalidate: () => {
      if (cacheKey) {
        apiCache.invalidate(cacheKey);
      }
    },
    clear: apiCache.clear,
  };
  
  // Return all the necessary data and functions
  return {
    data,
    loading,
    error,
    execute,
    paginated,
    cache,
    execCount,
  };
}

/**
 * Custom hook for API calls with automatic parameter binding
 * Useful for components that need a simpler interface
 */
export function useApiWithParams(apiFn, params = {}, options = {}) {
  const api = useApi(
    async () => apiFn(params),
    {
      ...options,
      deps: [JSON.stringify(params), ...(options.deps || [])],
    }
  );
  
  return api;
}

/**
 * Hook for tracking long-running operations like code evaluation
 */
export function useLongOperation(operationFn, options = {}) {
  const {
    pollingInterval = 1000,
    maxAttempts = 30,
    successCondition = (result) => result?.status === 'completed',
    onComplete = null,
  } = options;
  
  const [status, setStatus] = useState('idle'); // idle, running, completed, failed
  const [result, setResult] = useState(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  
  const attemptsRef = useRef(0);
  const operationIdRef = useRef(null);
  
  const start = useCallback(async (...args) => {
    try {
      setStatus('running');
      setError(null);
      attemptsRef.current = 0;
      
      // Start the operation and get the operation ID
      const operation = await operationFn(...args);
      operationIdRef.current = operation.id;
      
      return operation;
    } catch (err) {
      setStatus('failed');
      setError(err);
      throw err;
    }
  }, [operationFn]);
  
  const checkStatus = useCallback(async (statusFn) => {
    if (!operationIdRef.current || status !== 'running') return;
    
    try {
      attemptsRef.current += 1;
      
      const statusResult = await statusFn(operationIdRef.current);
      setProgress(statusResult.progress || 0);
      
      if (successCondition(statusResult)) {
        setStatus('completed');
        setResult(statusResult);
        if (onComplete) {
          onComplete(statusResult);
        }
      } else if (attemptsRef.current >= maxAttempts) {
        throw new Error('Operation timed out');
      }
    } catch (err) {
      setStatus('failed');
      setError(err);
    }
  }, [maxAttempts, onComplete, status, successCondition]);
  
  const cancel = useCallback(() => {
    if (status === 'running') {
      setStatus('idle');
      operationIdRef.current = null;
      attemptsRef.current = 0;
    }
  }, [status]);
  
  return {
    status,
    result,
    progress,
    error,
    start,
    checkStatus,
    cancel,
    isIdle: status === 'idle',
    isRunning: status === 'running',
    isCompleted: status === 'completed',
    isFailed: status === 'failed',
  };
}

/**
 * Hook for batch operations on multiple items
 */
export function useBatchOperation(itemOperationFn, options = {}) {
  const {
    concurrency = 3,
    onItemComplete = null,
    onAllComplete = null,
  } = options;
  
  const [items, setItems] = useState([]);
  const [completed, setCompleted] = useState([]);
  const [failed, setFailed] = useState([]);
  const [inProgress, setInProgress] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  
  const queue = useRef([]);
  
  const processQueue = useCallback(async () => {
    if (queue.current.length === 0 || inProgress.length >= concurrency) {
      return;
    }
    
    const item = queue.current.shift();
    setInProgress(prev => [...prev, item]);
    
    try {
      const result = await itemOperationFn(item);
      setCompleted(prev => [...prev, { item, result }]);
      setInProgress(prev => prev.filter(i => i !== item));
      
      if (onItemComplete) {
        onItemComplete(item, result, null);
      }
    } catch (error) {
      setFailed(prev => [...prev, { item, error }]);
      setInProgress(prev => prev.filter(i => i !== item));
      
      if (onItemComplete) {
        onItemComplete(item, null, error);
      }
    }
    
    // Continue processing the queue
    processQueue();
    
    // Check if we're done
    if (queue.current.length === 0 && inProgress.length === 0) {
      setIsRunning(false);
      if (onAllComplete) {
        onAllComplete(completed, failed);
      }
    }
  }, [itemOperationFn, concurrency, inProgress, onItemComplete, onAllComplete, completed, failed]);
  
  const startBatch = useCallback((itemsToProcess) => {
    setItems(itemsToProcess);
    setCompleted([]);
    setFailed([]);
    setInProgress([]);
    setIsRunning(true);
    
    queue.current = [...itemsToProcess];
    
    // Start processing up to concurrency items
    for (let i = 0; i < Math.min(concurrency, queue.current.length); i++) {
      processQueue();
    }
  }, [concurrency, processQueue]);
  
  const cancel = useCallback(() => {
    queue.current = [];
    setIsRunning(false);
  }, []);
  
  const stats = {
    total: items.length,
    completed: completed.length,
    failed: failed.length,
    inProgress: inProgress.length,
    queued: queue.current.length,
    percentage: items.length ? Math.round((completed.length + failed.length) / items.length * 100) : 0,
  };
  
  return {
    startBatch,
    cancel,
    isRunning,
    completed,
    failed,
    inProgress,
    stats,
  };
}

export default useApi;