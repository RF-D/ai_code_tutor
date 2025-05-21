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
    <div className="flex min-h-full w-full transition-colors duration-300 ease-in-out">
      <Navigation 
        isMobileOpen={isMobileNavOpen} 
        toggleMobileNav={toggleMobileNav} 
      />
      
      <main className="ml-[250px] min-h-screen flex-1 w-[calc(100%-250px)] transition-all duration-300 p-4 md:p-6 lg:p-8
                      dark:bg-[var(--bg-primary)] dark:text-[var(--text-primary)]
                      sm:ml-[250px] sm:w-[calc(100%-250px)]
                      md:ml-[250px] md:w-[calc(100%-250px)]
                      lg:ml-[250px] lg:w-[calc(100%-250px)]
                      max-md:ml-0 max-md:w-full">
        {/* Global notification area could go here */}
        <div className="max-w-[var(--container-xl)] mx-auto p-4 animate-[fadeIn_var(--transition-normal)_forwards]">
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default MainLayout;