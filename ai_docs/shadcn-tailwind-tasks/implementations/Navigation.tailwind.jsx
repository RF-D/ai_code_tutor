import React, { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { getNavigationRoutes } from '../../routes';
import LanguageSelector from './LanguageSelector';
import { useTheme } from '../../context/ThemeContext';

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
        className="fixed top-4 left-4 z-[var(--z-index-sticky)] cursor-pointer p-2 rounded-md bg-background 
                  shadow-sm border border-[var(--border-color)] transition-colors duration-300 hidden max-md:block"
        onClick={toggleMobileNav}
      >
        <span className={`relative w-6 h-[18px] inline-block ${isMobileOpen ? 'open' : ''}`}>
          <span className={`absolute w-full h-0.5 bg-[var(--text-primary)] left-0 transition-all duration-300 
                         ${isMobileOpen ? 'rotate-45 top-2' : 'top-0'}`}></span>
          <span className={`absolute w-full h-0.5 bg-[var(--text-primary)] left-0 top-2 transition-all duration-300 
                         ${isMobileOpen ? 'opacity-0' : ''}`}></span>
          <span className={`absolute w-full h-0.5 bg-[var(--text-primary)] left-0 transition-all duration-300 
                         ${isMobileOpen ? '-rotate-45 top-2' : 'top-4'}`}></span>
        </span>
      </div>

      {/* Navigation sidebar */}
      <nav 
        className={`w-[250px] h-screen fixed top-0 left-0 bg-muted border-r border-[var(--border-color)] 
                  flex flex-col transition-transform duration-300 ease-[var(--transition-ease)] z-[var(--z-index-fixed)] shadow-sm
                  ${isMobileOpen ? 'translate-x-0' : 'max-md:-translate-x-full'}`}
      >
        <div className="p-4 border-b border-[var(--border-color)] flex items-center justify-between">
          <div className="flex items-center">
            <span className="mr-2 text-primary text-xl">📚</span>
            <h2 className="m-0 text-xl font-semibold text-[var(--text-primary)]">AI Code Tutor</h2>
          </div>
        </div>

        <ul className="list-none p-0 m-0 flex-1 overflow-y-auto">
          {navRoutes.map((route) => (
            <li key={route.path} className="w-full">
              <NavLink
                to={route.path}
                className={({ isActive }) => 
                  `flex items-center py-3 px-4 text-[var(--text-secondary)] transition-all duration-[var(--transition-fast)] 
                  border-l-[3px] border-transparent hover:bg-accent hover:text-[var(--text-primary)] hover:no-underline
                  ${isActive ? 'bg-accent text-primary font-medium border-l-[3px] border-primary' : ''}`
                }
              >
                {/* Icon placeholder - could be replaced with actual icons */}
                <span className="mr-2 text-lg flex items-center justify-center w-5">
                  {route.meta.icon || (
                    route.path === '/' ? '🏠' : 
                    route.path.includes('playground') ? '💻' : 
                    route.path.includes('questions') ? '📝' : 
                    route.path.includes('settings') ? '⚙️' : '📄'
                  )}
                </span>
                <span className="flex-1">{route.meta.title}</span>
              </NavLink>
            </li>
          ))}
        </ul>

        <div className="p-4 border-t border-[var(--border-color)] flex flex-col gap-3">
          <div className="flex items-center justify-between mb-2">
            <span className="mr-2 text-sm text-[var(--text-secondary)]">
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
              <span className="absolute cursor-pointer top-0 left-0 right-0 bottom-0 bg-accent transition-[var(--transition-normal)] 
                              rounded-full before:absolute before:content-[''] before:h-4 before:w-4 before:left-0.5 before:bottom-0.5 
                              before:bg-background before:transition-[var(--transition-normal)] before:rounded-full
                              checked:bg-primary peer-checked:before:translate-x-5"></span>
            </label>
          </div>
          
          <LanguageSelector />
          {/* User info could go here */}
        </div>
      </nav>

      {/* Overlay for mobile */}
      {isMobileOpen && (
        <div 
          className="fixed top-0 left-0 right-0 bottom-0 bg-black/50 z-[var(--z-index-dropdown)] md:hidden"
          onClick={toggleMobileNav}
        ></div>
      )}
    </>
  );
};

export default Navigation;