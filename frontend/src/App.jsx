import React from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import routes from './routes.jsx';
import { AppProvider } from './context/AppContext.jsx';
import { LanguageProvider } from './context/LanguageContext.jsx';
import { QuestionProvider } from './context/QuestionContext.jsx';
import { ThemeProvider } from './context/ThemeContext.jsx';
import './App.css';

// Create router config from routes array
const router = createBrowserRouter(routes);

function App() {
  return (
    <ThemeProvider>
      <AppProvider>
        <LanguageProvider>
          <QuestionProvider>
            <RouterProvider router={router} />
          </QuestionProvider>
        </LanguageProvider>
      </AppProvider>
    </ThemeProvider>
  );
}

export default App;