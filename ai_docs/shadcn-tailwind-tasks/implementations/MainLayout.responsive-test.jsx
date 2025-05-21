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
      
      <main className="ml-[250px] min-h-screen flex-1 w-[calc(100%-250px)] transition-all duration-300 p-4
                      dark:bg-[var(--bg-primary)] dark:text-[var(--text-primary)]
                      max-md:ml-0 max-md:w-full">
        {/* Responsive Debug Information */}
        <div className="fixed bottom-4 right-4 z-50 bg-primary text-white p-2 rounded shadow-md">
          <div className="sm:hidden">Current Breakpoint: xs (below 640px)</div>
          <div className="hidden sm:block md:hidden">Current Breakpoint: sm (640px - 767px)</div>
          <div className="hidden md:block lg:hidden">Current Breakpoint: md (768px - 1023px)</div>
          <div className="hidden lg:block xl:hidden">Current Breakpoint: lg (1024px - 1279px)</div>
          <div className="hidden xl:block 2xl:hidden">Current Breakpoint: xl (1280px - 1535px)</div>
          <div className="hidden 2xl:block">Current Breakpoint: 2xl (1536px+)</div>
        </div>

        {/* Main Content */}
        <div className="max-w-[var(--container-xl)] mx-auto p-4 animate-[fadeIn_var(--transition-normal)_forwards]
                        sm:p-5 md:p-6 lg:p-8">
          <div className="bg-muted p-4 mb-4 rounded-lg shadow-sm border border-border">
            <h2 className="text-xl font-bold mb-2">Responsive Testing</h2>
            <p className="mb-2">This layout shows different styles at different breakpoints:</p>
            <ul className="list-disc pl-5 mb-4">
              <li>xs (below 640px): Full-width content, navigation hidden</li>
              <li>sm (640px - 767px): Navigation shows on toggle</li>
              <li>md (768px - 1023px): Side navigation visible, reduced padding</li>
              <li>lg (1024px+): Side navigation visible, increased padding</li>
            </ul>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              <div className="bg-card p-4 rounded border border-border">Item 1</div>
              <div className="bg-card p-4 rounded border border-border">Item 2</div>
              <div className="bg-card p-4 rounded border border-border">Item 3</div>
              <div className="bg-card p-4 rounded border border-border">Item 4</div>
            </div>
          </div>
          <Outlet />
        </div>
      </main>
    </div>
  );
};

export default MainLayout;