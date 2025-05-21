import React, { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { FaLaptopCode, FaQuestionCircle, FaCog, FaHome } from 'react-icons/fa';
import { getNavigationRoutes } from '../../routes';
import LanguageSelector from './LanguageSelector';
import { useTheme } from '../../context/ThemeContext';
import useResponsive from '../../hooks/useResponsive';

/**
 * Navigation component providing sidebar navigation for the application
 * Enhanced with responsive design using Tailwind CSS and useResponsive hook
 */
const Navigation = ({ isMobileOpen, toggleMobileNav }) => {
  const [navRoutes, setNavRoutes] = useState([]);
  const location = useLocation();
  const { theme, toggleTheme, isDarkMode } = useTheme();
  const { below, above, orientation, breakpoint } = useResponsive();

  const iconMap = {
    code: <FaLaptopCode />,
    question: <FaQuestionCircle />,
    settings: <FaCog />,
    home: <FaHome />
  };
  
  // Determine if we should use compact view on mobile
  const useCompactView = below('md');
  
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
      {/* Mobile navigation toggle - visible on small screens */}
      <button
        aria-label="Toggle navigation menu"
        className={`fixed top-3 left-3 z-50 md:hidden cursor-pointer p-2
                   rounded-md bg-background-primary shadow-md border border-border 
                   transition-colors duration-300 hover:bg-background-secondary
                   focus:outline-none focus:ring-2 focus:ring-primary focus:ring-opacity-50`}
        onClick={toggleMobileNav}
      >
        <span className="w-6 h-5 relative inline-block">
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
      </button>

      {/* Navigation sidebar */}
      <nav 
        className={`w-[250px] h-screen fixed top-0 left-0 bg-background-secondary 
                    border-r border-border flex flex-col transition-all 
                    duration-300 ease-in-out z-40 shadow-md
                    md:translate-x-0 
                    ${isMobileOpen ? 'translate-x-0' : '-translate-x-full'}`}
      >
        <div className="p-4 border-b border-border flex items-center justify-between">
          <div className="flex items-center">
            <span className="mr-2 text-primary text-xl" aria-hidden="true">📚</span>
            <h2 className="m-0 text-xl font-semibold text-text-primary truncate">
              {useCompactView ? 'AI Tutor' : 'AI Code Tutor'}
            </h2>
          </div>

          {/* Close button - only visible on mobile when menu is open */}
          {below('md') && (
            <button
              aria-label="Close navigation menu"
              className="md:hidden text-text-secondary p-1 rounded-full 
                        hover:bg-background-tertiary transition-colors"
              onClick={toggleMobileNav}
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
            </button>
          )}
        </div>

        {/* Navigation links */}
        <ul className="list-none p-0 m-0 flex-1 overflow-y-auto">
          {navRoutes.map((route) => (
            <li key={route.path} className="w-full">
              <NavLink
                to={route.path}
                className={({ isActive }) => 
                  `flex items-center py-3 px-4 text-text-secondary no-underline 
                   transition-all duration-200 border-l-[3px] border-transparent
                   hover:bg-background-tertiary hover:no-underline hover:text-text-primary
                   focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary
                   ${isActive ? 'bg-background-tertiary text-primary font-medium border-l-[3px] border-primary' : ''}`
                }
                onClick={useCompactView ? toggleMobileNav : undefined}
              >
                <span className="mr-2 text-lg flex items-center justify-center w-5" aria-hidden="true">
                  {iconMap[route.meta.icon] || (
                    route.path === '/' ? <FaHome /> :
                    route.path.includes('playground') ? <FaLaptopCode /> :
                    route.path.includes('questions') ? <FaQuestionCircle /> :
                    route.path.includes('settings') ? <FaCog /> : '📄'
                  )}
                </span>
                <span className="flex-1 truncate">{route.meta.title}</span>
              </NavLink>
            </li>
          ))}
        </ul>

        {/* Footer with theme toggle and language selector */}
        <div className="p-4 border-t border-border flex flex-col gap-3">
          <div className="flex items-center justify-between mb-2">
            <span className="mr-2 text-sm text-text-secondary">
              {isDarkMode ? 'Dark Mode' : 'Light Mode'}
            </span>
            <label className="relative inline-block w-10 h-5">
              <input 
                type="checkbox" 
                className="sr-only"
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
          
          {/* Only show on mobile - version indicator in footer */}
          {useCompactView && (
            <div className="mt-2 text-xs text-text-secondary opacity-70 text-center">
              v1.0.0
            </div>
          )}
        </div>
      </nav>

      {/* Overlay for mobile - visible when menu is open */}
      {isMobileOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-30 md:hidden"
          onClick={toggleMobileNav}
          aria-hidden="true"
        ></div>
      )}
    </>
  );
};

export default Navigation;
