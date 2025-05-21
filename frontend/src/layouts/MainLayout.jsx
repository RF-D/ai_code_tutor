import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import Navigation from '../components/common/Navigation';
import { useTheme } from '../context/ThemeContext';

const MainLayout = () => {
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);
  const { theme } = useTheme();

  const toggleMobileNav = () => {
    setIsMobileNavOpen(!isMobileNavOpen);
  };

  return (
    <div className="app-container theme-transition">
      <Navigation 
        isMobileOpen={isMobileNavOpen} 
        toggleMobileNav={toggleMobileNav} 
      />
      
      <main className="main-content">
        {/* Global notification area could go here */}
        <div className="page-container animate-fade-in">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default MainLayout;