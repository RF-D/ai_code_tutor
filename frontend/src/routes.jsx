import React, { lazy, Suspense } from 'react';
import { Navigate } from 'react-router-dom';

// Error boundary for code-split components
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    console.error('Route error boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center w-full h-full min-h-[400px]">
          <div className="text-center">
            <h2 className="text-xl font-semibold mb-2">Something went wrong</h2>
            <p className="text-text-secondary mb-4">Failed to load the page</p>
            <button 
              className="px-4 py-2 bg-primary text-white rounded"
              onClick={() => this.setState({ hasError: false })}
            >
              Try again
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Layouts
import MainLayout from './layouts/MainLayout.jsx';

// Loading component for Suspense fallback
const PageLoader = () => (
  <div className="flex items-center justify-center w-full h-full min-h-[400px]">
    <div className="text-center">
      <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto"></div>
      <p className="mt-3 text-text-secondary">Loading...</p>
    </div>
  </div>
);

// Enhanced lazy loading with chunking and prefetching
const PlaygroundPage = lazy(() => import(/* webpackChunkName: "playground" */ './pages/PlaygroundPage.jsx'));
const QuestionsPage = lazy(() => import(/* webpackChunkName: "questions" */ './pages/QuestionsPage.jsx'));
const SettingsPage = lazy(() => import(/* webpackChunkName: "settings" */ './pages/SettingsPage.jsx'));

// Prefetch component for preloading important routes
const Prefetch = ({ children }) => {
  React.useEffect(() => {
    // Prefetch the most common routes after initial page load
    const prefetchPlayground = import('./pages/PlaygroundPage.jsx');
    const prefetchQuestions = import('./pages/QuestionsPage.jsx');
    
    // Intentionally not awaiting these promises
  }, []);
  
  return <>{children}</>;
};

// Route definitions with metadata
const routes = [
  {
    path: '/',
    element: <MainLayout />,
    children: [
      { path: '/', element: <Navigate to="/playground" replace /> },
      {
        path: 'playground',
        element: (
          <ErrorBoundary>
            <Suspense fallback={<PageLoader />}>
              <PlaygroundPage />
            </Suspense>
          </ErrorBoundary>
        ),
        meta: {
          title: 'Code Playground',
          icon: 'code',
          requiresAuth: false
        }
      },
      {
        path: 'questions',
        element: (
          <ErrorBoundary>
            <Suspense fallback={<PageLoader />}>
              <QuestionsPage />
            </Suspense>
          </ErrorBoundary>
        ),
        meta: {
          title: 'Practice Questions',
          icon: 'question',
          requiresAuth: false
        }
      },
      {
        path: 'settings',
        element: (
          <ErrorBoundary>
            <Suspense fallback={<PageLoader />}>
              <SettingsPage />
            </Suspense>
          </ErrorBoundary>
        ),
        meta: {
          title: 'Settings',
          icon: 'settings',
          requiresAuth: false
        }
      },
      // Add a catch-all route that redirects to playground
      { path: '*', element: <Navigate to="/playground" replace /> }
    ]
  }
];

// Helper function to get routes for navigation
export const getNavigationRoutes = () => {
  // Get only the routes we want to show in navigation
  return routes[0].children.filter(route => route.meta && !route.meta.hideInNav);
};

// Wrap the app in Prefetch to preload important routes
export const AppWithPrefetch = ({ children }) => (
  <Prefetch>{children}</Prefetch>
);

export default routes;
