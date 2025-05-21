import React, { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { getNavigationRoutes } from '../../routes';
import LanguageSelector from './LanguageSelector';
import { useTheme } from '../../context/ThemeContext';

/**
 * Navigation component providing sidebar navigation for the application
 * Updated to use Tailwind CSS for styling
 */
const Navigation = ({ isMobileOpen, toggleMobileNav }) => {
  const [navRoutes, setNavRoutes] = useState([]);
  const location = useLocation();
  const { theme, toggleTheme, isDarkMode } = useTheme();
  
  // Get navigation routes on component mount
  useEffect(() => {
    setNavRoutes(getNavigationRoutes());
  }, []);

  // Close mobile nav when route changes
  useEffect(() => {
    if (isMobileOpen) {
      toggleMobileNav();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname]);

  return (
    <>
      {/* Mobile navigation toggle */}
      <div 
        className="fixed top-4 left-4 z-40 hidden sm:block cursor-pointer p-2 
                   rounded-md bg-background-primary shadow-sm border border-border 
                   transition-colors duration-300"
        onClick={toggleMobileNav}
      >
        <span className="w-6 h-[18px] relative inline-block">
          <span 
            className={`w-full h-0.5 bg-text-primary absolute left-0 transition-all duration-300 
                        ${isMobileOpen ? 'top-2 rotate-45' : 'top-0'}`}
          ></span>
          <span 
            className={`w-full h-0.5 bg-text-primary absolute left-0 top-2 transition-all duration-300 
                        ${isMobileOpen ? 'opacity-0' : 'opacity-100'}`}
          ></span>
          <span 
            className={`w-full h-0.5 bg-text-primary absolute left-0 transition-all duration-300 
                        ${isMobileOpen ? 'top-2 -rotate-45' : 'top-4'}`}
          ></span>
        </span>
      </div>

      {/* Navigation sidebar */}
      <nav 
        className={`w-[250px] h-screen fixed top-0 left-0 bg-background-secondary 
                    border-r border-border flex flex-col transition-transform 
                    duration-300 ease-in-out z-40 shadow-sm
                    sm:transform ${isMobileOpen ? 'sm:translate-x-0' : 'sm:-translate-x-full'}`}
      >
        <div className="p-4 border-b border-border flex items-center justify-between">
          <div className="flex items-center">
            <span className="mr-2 text-primary text-xl">📚</span>
            <h2 className="m-0 text-xl font-semibold text-text-primary">AI Code Tutor</h2>
          </div>
        </div>

        <ul className="list-none p-0 m-0 flex-1 overflow-y-auto">
          {navRoutes.map((route) => (
            <li key={route.path} className="w-full">
              <NavLink
                to={route.path}
                className={({ isActive }) => 
                  `flex items-center py-3 px-4 text-text-secondary no-underline 
                   transition-all duration-200 border-l-[3px] border-transparent
                   hover:bg-background-tertiary hover:no-underline hover:text-text-primary
                   ${isActive ? 'bg-background-tertiary text-primary font-medium border-l-[3px] border-primary' : ''}`
                }
              >
                <span className="mr-2 text-lg flex items-center justify-center w-5">
                  {route.meta.icon || (
                    route.path === '/' ? '🏠' : 
                    route.path.includes('playground') ? '💻' : 
                    route.path.includes('questions') ? '📝' : 
                    route.path.includes('shadcn') ? '🎨' :
                    route.path.includes('settings') ? '⚙️' : '📄'
                  )}
                </span>
                <span className="flex-1">{route.meta.title}</span>
              </NavLink>
            </li>
          ))}
        </ul>

        <div className="p-4 border-t border-border flex flex-col gap-3">
          <div className="flex items-center justify-between mb-2">
            <span className="mr-2 text-sm text-text-secondary">
              {isDarkMode ? 'Dark Mode' : 'Light Mode'}
            </span>
            <label className="relative inline-block w-10 h-5">
              <input 
                type="checkbox" 
                className="opacity-0 w-0 h-0"
                checked={isDarkMode}
                onChange={toggleTheme}
                aria-label="Toggle dark mode"
              />
              <span 
                className={`absolute cursor-pointer top-0 left-0 right-0 bottom-0 
                           ${isDarkMode ? 'bg-primary' : 'bg-background-tertiary'} 
                           transition-colors duration-300 rounded-full
                           before:absolute before:content-[''] before:h-4 before:w-4 
                           before:left-0.5 before:bottom-0.5 before:bg-background-primary 
                           before:rounded-full before:transition-transform before:duration-300
                           ${isDarkMode ? 'before:translate-x-5' : ''}`}
              ></span>
            </label>
          </div>
          
          <LanguageSelector />
        </div>
      </nav>

      {/* Overlay for mobile */}
      {isMobileOpen && (
        <div 
          className="fixed top-0 left-0 right-0 bottom-0 bg-black/50 z-30 sm:block hidden" 
          onClick={toggleMobileNav}
        ></div>
      )}
    </>
  );
};

export default Navigation;