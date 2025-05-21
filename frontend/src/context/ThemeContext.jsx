import React, { createContext, useState, useEffect, useContext } from 'react';

// Create theme context
const ThemeContext = createContext();

// Define available themes
export const THEMES = {
  LIGHT: 'light',
  DARK: 'dark',
};

// Theme provider component
export const ThemeProvider = ({ children }) => {
  // Initialize theme state from localStorage or system preference
  const [theme, setTheme] = useState(() => {
    // Check if there's a saved theme in localStorage
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme) {
      return savedTheme;
    }
    
    // Check system preference for dark mode
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return THEMES.DARK;
    }
    
    // Default to light theme
    return THEMES.LIGHT;
  });

  // Function to toggle between light and dark themes
  const toggleTheme = () => {
    setTheme(prevTheme => 
      prevTheme === THEMES.LIGHT ? THEMES.DARK : THEMES.LIGHT
    );
  };

  // Function to set a specific theme
  const setThemeMode = (newTheme) => {
    if (Object.values(THEMES).includes(newTheme)) {
      setTheme(newTheme);
    } else {
      console.error(`Invalid theme: ${newTheme}. Available themes: ${Object.values(THEMES).join(', ')}`);
    }
  };

  // Update document attributes and localStorage when theme changes
  useEffect(() => {
    // Update data-theme attribute on document element
    document.documentElement.setAttribute('data-theme', theme);
    
    // Save theme preference to localStorage
    localStorage.setItem('theme', theme);
  }, [theme]);

  // Listen for system theme preference changes
  useEffect(() => {
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    
    const handleChange = (e) => {
      // Only update if the user hasn't explicitly set a theme
      if (!localStorage.getItem('theme')) {
        setTheme(e.matches ? THEMES.DARK : THEMES.LIGHT);
      }
    };
    
    // Add event listener for theme preference changes
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleChange);
    } else {
      // Fallback for older browsers
      mediaQuery.addListener(handleChange);
    }
    
    // Cleanup function
    return () => {
      if (mediaQuery.removeEventListener) {
        mediaQuery.removeEventListener('change', handleChange);
      } else {
        // Fallback for older browsers
        mediaQuery.removeListener(handleChange);
      }
    };
  }, []);

  // Value object to be provided by the context
  const contextValue = {
    theme,
    toggleTheme,
    setTheme: setThemeMode,
    isDarkMode: theme === THEMES.DARK,
    isLightMode: theme === THEMES.LIGHT,
  };

  return (
    <ThemeContext.Provider value={contextValue}>
      {children}
    </ThemeContext.Provider>
  );
};

// Custom hook for using the theme context
export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

export default ThemeContext;