import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Navigation from '../components/common/Navigation';
import { useTheme } from '../context/ThemeContext';
import useResponsive from '../hooks/useResponsive';

/**
 * MainLayout component that provides the main application layout structure
 * with navigation and content area. Enhanced with responsive design.
 */
const MainLayout = () => {
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);
  const { theme } = useTheme();
  const { below, above, orientation } = useResponsive();

  // Whether we should use the mobile layout
  const isMobile = below('md');

  const toggleMobileNav = () => {
    setIsMobileNavOpen(!isMobileNavOpen);
  };

  // Prevent scrolling when mobile nav is open
  if (typeof document !== 'undefined') {
    if (isMobileNavOpen && isMobile) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
  }

  return (
    <div className="flex min-h-screen w-full transition-colors duration-300 overflow-x-hidden bg-background-primary">
      <Navigation 
        isMobileOpen={isMobileNavOpen} 
        toggleMobileNav={toggleMobileNav} 
      />
      
      <main 
        className={`
          min-h-screen flex-1 transition-all duration-300
          md:ml-[250px] md:w-[calc(100%-250px)]
          w-full
          p-3 sm:p-4 md:p-6
        `}
      >
        {/* Page content with spacing that adapts to screen size */}
        <div className={`
          max-w-7xl mx-auto 
          ${isMobile ? 'pt-12' : 'pt-4'} 
          px-2 sm:px-4 
          pb-6 
          animate-[fadeIn_0.3s_forwards]
        `}>
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default MainLayout;