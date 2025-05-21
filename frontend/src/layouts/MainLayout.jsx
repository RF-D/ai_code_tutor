import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Navigation from '../components/common/Navigation';
import { useTheme } from '../context/ThemeContext';

/**
 * MainLayout component that provides the main application layout structure
 * with navigation and content area. Updated to use Tailwind CSS.
 */
const MainLayout = () => {
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);
  const { theme } = useTheme();

  const toggleMobileNav = () => {
    setIsMobileNavOpen(!isMobileNavOpen);
  };

  return (
    <div className="flex min-h-full w-full transition-colors duration-300">
      <Navigation 
        isMobileOpen={isMobileNavOpen} 
        toggleMobileNav={toggleMobileNav} 
      />
      
      <main className="ml-[250px] min-h-screen flex-1 w-[calc(100%-250px)] transition-all duration-300 p-4 md:p-6 
                       sm:ml-0 sm:w-full">
        <div className="max-w-7xl mx-auto p-4 animate-[fadeIn_0.3s_forwards]">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default MainLayout;