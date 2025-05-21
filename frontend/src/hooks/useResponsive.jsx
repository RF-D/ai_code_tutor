import { useState, useEffect, useMemo } from 'react';

/**
 * A custom hook for responsive design that provides information about
 * the current viewport size based on common breakpoints.
 * 
 * @returns {Object} An object containing responsive design information:
 *   - isMobile: boolean, true if viewport width is less than 576px
 *   - isTablet: boolean, true if viewport width is between 576px and 991px
 *   - isDesktop: boolean, true if viewport width is 992px or more
 *   - width: number, the current viewport width
 *   - height: number, the current viewport height
 *   - breakpoint: string, current breakpoint name ('xs', 'sm', 'md', 'lg', 'xl', '2xl')
 *   - orientation: string, 'portrait' or 'landscape'
 *   - below: function, checks if current width is below a specific breakpoint
 *   - above: function, checks if current width is above a specific breakpoint
 *   - between: function, checks if current width is between two breakpoints
 */
const useResponsive = () => {
  // Define breakpoint values
  const breakpoints = useMemo(() => ({
    xs: 0,
    sm: 576,
    md: 768,
    lg: 992,
    xl: 1200,
    '2xl': 1400
  }), []);

  // Initial state with default values
  const [responsive, setResponsive] = useState({
    isMobile: false,
    isTablet: false, 
    isDesktop: true,
    width: typeof window !== 'undefined' ? window.innerWidth : 1200,
    height: typeof window !== 'undefined' ? window.innerHeight : 800,
    breakpoint: 'lg',
    orientation: 'landscape'
  });

  useEffect(() => {
    // Function to determine the current breakpoint
    const getBreakpoint = (width) => {
      if (width < breakpoints.sm) return 'xs';
      if (width < breakpoints.md) return 'sm';
      if (width < breakpoints.lg) return 'md';
      if (width < breakpoints.xl) return 'lg';
      if (width < breakpoints['2xl']) return 'xl';
      return '2xl';
    };

    // Handler to update responsive state when resize occurs
    const handleResize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      const breakpoint = getBreakpoint(width);
      const orientation = height > width ? 'portrait' : 'landscape';
      
      setResponsive({
        isMobile: width < breakpoints.sm,
        isTablet: width >= breakpoints.sm && width < breakpoints.lg,
        isDesktop: width >= breakpoints.lg,
        width,
        height,
        breakpoint,
        orientation
      });
    };

    // Handler for device orientation change
    const handleOrientationChange = () => {
      handleResize();
    };

    // Register the event listeners
    if (typeof window !== 'undefined') {
      // Initial calculation
      handleResize();
      
      // Add event listeners
      window.addEventListener('resize', handleResize);
      window.addEventListener('orientationchange', handleOrientationChange);
      
      // Clean up event listeners
      return () => {
        window.removeEventListener('resize', handleResize);
        window.removeEventListener('orientationchange', handleOrientationChange);
      };
    }
  }, [breakpoints]);

  // Helper functions to check breakpoint conditions
  const below = useMemo(() => 
    (breakpoint) => responsive.width < breakpoints[breakpoint],
  [responsive.width, breakpoints]);

  const above = useMemo(() => 
    (breakpoint) => responsive.width >= breakpoints[breakpoint],
  [responsive.width, breakpoints]);

  const between = useMemo(() => 
    (minBreakpoint, maxBreakpoint) => 
      responsive.width >= breakpoints[minBreakpoint] && 
      responsive.width < breakpoints[maxBreakpoint],
  [responsive.width, breakpoints]);

  return {
    ...responsive,
    below,
    above,
    between,
    breakpoints
  };
};

export default useResponsive;