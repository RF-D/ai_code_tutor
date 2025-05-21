import React, { useState, useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { getNavigationRoutes } from '../../routes';
import LanguageSelector from './LanguageSelector';
import { useTheme } from '../../context/ThemeContext';
import styles from '../../styles/components/Navigation.module.css';

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
      <div className={styles.mobileNavToggle} onClick={toggleMobileNav}>
        <span className={`${styles.hamburger} ${isMobileOpen ? styles.open : ''}`}>
          <span></span>
          <span></span>
          <span></span>
        </span>
      </div>

      {/* Navigation sidebar */}
      <nav className={`${styles.navigation} ${isMobileOpen ? styles.mobileOpen : ''}`}>
        <div className={styles.navHeader}>
          <div className={styles.logoContainer}>
            <span className={styles.logoIcon}>📚</span>
            <h2 className={styles.navTitle}>AI Code Tutor</h2>
          </div>
        </div>

        <ul className={styles.navLinks}>
          {navRoutes.map((route) => (
            <li key={route.path} className={styles.navItem}>
              <NavLink
                to={route.path}
                className={({ isActive }) => 
                  isActive ? `${styles.navLink} ${styles.active}` : styles.navLink
                }
              >
                {/* Icon placeholder - could be replaced with actual icons */}
                <span className={styles.navIcon}>
                  {route.meta.icon || (
                    route.path === '/' ? '🏠' : 
                    route.path.includes('playground') ? '💻' : 
                    route.path.includes('questions') ? '📝' : 
                    route.path.includes('settings') ? '⚙️' : '📄'
                  )}
                </span>
                <span className={styles.navLabel}>{route.meta.title}</span>
              </NavLink>
            </li>
          ))}
        </ul>

        <div className={styles.navFooter}>
          <div className={styles.themeSwitcher}>
            <span className={styles.toggleLabel}>
              {isDarkMode ? 'Dark Mode' : 'Light Mode'}
            </span>
            <label className={styles.toggleSwitch}>
              <input 
                type="checkbox" 
                className={styles.toggleInput}
                checked={isDarkMode}
                onChange={toggleTheme}
                aria-label="Toggle dark mode"
              />
              <span className={styles.toggleSlider}></span>
            </label>
          </div>
          
          <LanguageSelector />
          {/* User info could go here */}
        </div>
      </nav>

      {/* Overlay for mobile */}
      {isMobileOpen && (
        <div className={styles.navOverlay} onClick={toggleMobileNav}></div>
      )}
    </>
  );
};

export default Navigation;