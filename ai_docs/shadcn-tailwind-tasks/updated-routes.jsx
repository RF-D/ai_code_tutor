import React from 'react';
import { Navigate } from 'react-router-dom';

// Layouts
import MainLayout from './layouts/MainLayout.jsx';

// Pages
import PlaygroundPage from './pages/PlaygroundPage.jsx';
import QuestionsPage from './pages/QuestionsPage.jsx';
import SettingsPage from './pages/SettingsPage.jsx';
import ShadcnTestPage from './pages/ShadcnTestPage.jsx';

// Route definitions with metadata
const routes = [
  {
    path: '/',
    element: <MainLayout />,
    children: [
      { path: '/', element: <Navigate to="/playground" replace /> },
      {
        path: 'playground',
        element: <PlaygroundPage />,
        meta: {
          title: 'Code Playground',
          icon: 'code',
          requiresAuth: false
        }
      },
      {
        path: 'questions',
        element: <QuestionsPage />,
        meta: {
          title: 'Practice Questions',
          icon: 'question',
          requiresAuth: false
        }
      },
      {
        path: 'settings',
        element: <SettingsPage />,
        meta: {
          title: 'Settings',
          icon: 'settings',
          requiresAuth: false
        }
      },
      {
        path: 'shadcn-test',
        element: <ShadcnTestPage />,
        meta: {
          title: 'Shadcn/UI Test',
          icon: 'palette',
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

export default routes;