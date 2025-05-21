import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';

/**
 * Creates a configured QueryClient with optimized caching settings
 * @returns {QueryClient} A configured query client instance
 */
const createQueryClient = () => {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // Setting reasonable cache times
        staleTime: 1000 * 60 * 5, // 5 minutes
        cacheTime: 1000 * 60 * 60, // 1 hour
        
        // Disable refetching on window focus in production
        refetchOnWindowFocus: process.env.NODE_ENV !== 'production',
        
        // Error handling
        retry: (failureCount, error) => {
          // Don't retry on 4xx responses
          if (error?.response?.status >= 400 && error?.response?.status < 500) {
            return false;
          }
          // Retry other errors up to 2 times
          return failureCount < 2;
        },
        
        // Performance optimization
        keepPreviousData: true,
        // Don't refetch on mount if data is still fresh
        refetchOnMount: 'always',
        
        // For better UX
        suspense: false,
      },
      mutations: {
        // Don't retry mutations by default
        retry: false,
      },
    },
  });
};

// Create a single instance of QueryClient
const queryClient = createQueryClient();

/**
 * QueryProvider component to wrap the app with React Query functionality
 */
export function QueryProvider({ children }) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {/* Only include Devtools in development */}
      {process.env.NODE_ENV !== 'production' && <ReactQueryDevtools />}
    </QueryClientProvider>
  );
}

export default QueryProvider;