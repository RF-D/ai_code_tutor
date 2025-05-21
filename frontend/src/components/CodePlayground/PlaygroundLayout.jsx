import React, { useState, useEffect, useMemo } from 'react';
import Split from 'react-split';
import CodePanel from './CodePanel';
import AssistantPanel from './AssistantPanel';
import ResultsPanel from './ResultsPanel';
import QuestionPanel from './QuestionPanel';
import useResponsive from '../../hooks/useResponsive';
import '../../styles/playground.css';

/**
 * PlaygroundLayout provides the main interface layout with resizable panels
 * for coding, getting assistance, viewing results, and reading questions.
 * 
 * Enhanced with Tailwind CSS for better responsive design and using the
 * useResponsive hook to determine the appropriate layout for different screen sizes.
 */
function PlaygroundLayout() {
  // Use the responsive hook to get current viewport information
  const { 
    isMobile, 
    isTablet, 
    isDesktop, 
    breakpoint, 
    orientation, 
    below, 
    above, 
    width,
    height 
  } = useResponsive();
  
  // Determine layout based on screen size and orientation
  const layout = useMemo(() => {
    if (below('sm')) return 'mobile';
    if (below('lg')) return 'tablet';
    return 'desktop';
  }, [below]);

  // Determine if we should use vertical orientation for the question/code split
  const useVerticalLayout = useMemo(() => {
    return below('lg') || (orientation === 'portrait' && below('xl'));
  }, [below, orientation]);
  
  // State for panel sizes with default values
  const [horizontalSizes, setHorizontalSizes] = useState(() => {
    const saved = localStorage.getItem('horizontalSizes');
    // Adjust default sizes based on layout
    if (saved) return JSON.parse(saved);
    if (layout === 'mobile') return [100];
    if (layout === 'tablet') return [65, 35];
    return [60, 40]; // Desktop: Left/Right default split percentages
  });
  
  const [leftHorizontalSizes, setLeftHorizontalSizes] = useState(() => {
    const saved = localStorage.getItem('leftHorizontalSizes');
    if (saved) return JSON.parse(saved);
    // Different ratios for different layouts
    if (useVerticalLayout) return [30, 70];
    return [40, 60]; // Question/Code default split percentages
  });
  
  const [rightVerticalSizes, setRightVerticalSizes] = useState(() => {
    const saved = localStorage.getItem('rightVerticalSizes');
    if (saved) return JSON.parse(saved);
    // Adjust based on layout
    if (layout === 'tablet' && orientation === 'portrait') return [60, 40];
    return [70, 30]; // Assistant/Results default split percentages
  });

  // Reset panel sizes when responsive breakpoint or orientation changes
  useEffect(() => {
    // Don't override user settings if present and screen size hasn't drastically changed
    if (layout === 'mobile') {
      setHorizontalSizes([100]);
      
      // Mobile phone in landscape mode
      if (orientation === 'landscape' && height < 500) {
        setLeftHorizontalSizes([20, 80]); // Less space for question, more for code
      } else {
        setLeftHorizontalSizes([30, 70]);
      }
      
      setRightVerticalSizes([60, 40]);
    } else if (layout === 'tablet') {
      setHorizontalSizes([65, 35]);
      
      if (orientation === 'portrait') {
        setLeftHorizontalSizes([30, 70]);
        setRightVerticalSizes([60, 40]);
      } else {
        setLeftHorizontalSizes([25, 75]);
        setRightVerticalSizes([65, 35]);
      }
    } else {
      // Desktop
      setHorizontalSizes([60, 40]);
      setLeftHorizontalSizes(useVerticalLayout ? [30, 70] : [40, 60]);
      setRightVerticalSizes([70, 30]);
    }
  }, [layout, orientation, useVerticalLayout, height]);

  // Save panel sizes to localStorage when they change
  useEffect(() => {
    localStorage.setItem('horizontalSizes', JSON.stringify(horizontalSizes));
  }, [horizontalSizes]);

  useEffect(() => {
    localStorage.setItem('leftHorizontalSizes', JSON.stringify(leftHorizontalSizes));
  }, [leftHorizontalSizes]);

  useEffect(() => {
    localStorage.setItem('rightVerticalSizes', JSON.stringify(rightVerticalSizes));
  }, [rightVerticalSizes]);

  // Handle code execution
  const [codeOutput, setCodeOutput] = useState('');
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionError, setExecutionError] = useState(null);
  const [executionTime, setExecutionTime] = useState(null);

  const handleCodeExecution = async (code) => {
    setIsExecuting(true);
    setExecutionError(null);
    setExecutionTime(null);
    
    try {
      // This would be replaced with an actual API call
      const response = await simulateCodeExecution(code);
      setCodeOutput(response.output);
      if (response.executionTime) {
        setExecutionTime(`${response.executionTime}ms`);
      }
    } catch (error) {
      setExecutionError(error.message);
      setCodeOutput('Error executing code. See error message for details.');
    } finally {
      setIsExecuting(false);
    }
  };

  // Simulate API call (placeholder for actual implementation)
  const simulateCodeExecution = (code) => {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          output: `Executed:\n${code}\n\n// This is simulated output. In a real implementation, this would show the actual results of code execution.`,
          executionTime: '0.34ms',
        });
      }, 1000);
    });
  };

  // For mobile view, render stacked layout
  if (layout === 'mobile') {
    return (
      <div className="playground-container w-full h-screen overflow-hidden bg-background-secondary text-text-primary">
        <div className="flex flex-col h-full w-full">
          {/* Mobile Layout: Stacked panels */}
          <div className={`w-full ${orientation === 'landscape' ? 'h-1/2' : 'h-[40%]'} overflow-hidden`}>
            <Split
              sizes={leftHorizontalSizes}
              minSize={orientation === 'landscape' ? 100 : 150}
              expandToMin={false}
              gutterSize={6}
              gutterAlign="center"
              direction="vertical"
              onDragEnd={setLeftHorizontalSizes}
              className="h-full"
            >
              <QuestionPanel />
              <CodePanel 
                onRunCode={handleCodeExecution} 
                isExecuting={isExecuting} 
              />
            </Split>
          </div>
          <div className={`w-full ${orientation === 'landscape' ? 'h-1/2' : 'h-[60%]'} overflow-hidden`}>
            <Split
              sizes={rightVerticalSizes}
              minSize={orientation === 'landscape' ? 80 : 100}
              expandToMin={false}
              gutterSize={6}
              gutterAlign="center"
              direction="vertical"
              onDragEnd={setRightVerticalSizes}
              className="h-full"
            >
              <AssistantPanel />
              <ResultsPanel 
                output={codeOutput} 
                error={executionError}
                isLoading={isExecuting}
                executionTime={executionTime}
              />
            </Split>
          </div>
        </div>
      </div>
    );
  }

  // For tablet and desktop, use appropriate layouts
  return (
    <div className="playground-container w-full h-screen overflow-hidden bg-background-secondary text-text-primary">
      <Split
        sizes={horizontalSizes}
        minSize={layout === 'tablet' ? 250 : 300}
        expandToMin={false}
        gutterSize={8}
        gutterAlign="center"
        direction="horizontal"
        className="playground-main"
        onDragEnd={setHorizontalSizes}
      >
        {/* Left section: Question + Code Editor */}
        <div className="left-section">
          <Split
            sizes={leftHorizontalSizes}
            minSize={layout === 'tablet' ? 120 : 150}
            expandToMin={false}
            gutterSize={8}
            gutterAlign="center"
            direction={useVerticalLayout ? "vertical" : "horizontal"}
            onDragEnd={setLeftHorizontalSizes}
          >
            <QuestionPanel />
            <CodePanel 
              onRunCode={handleCodeExecution} 
              isExecuting={isExecuting} 
            />
          </Split>
        </div>

        {/* Right section: Assistant + Results */}
        <div className="right-section">
          <Split
            sizes={rightVerticalSizes}
            minSize={layout === 'tablet' ? 80 : 100}
            expandToMin={false}
            gutterSize={8}
            gutterAlign="center"
            direction="vertical"
            onDragEnd={setRightVerticalSizes}
          >
            <AssistantPanel />
            <ResultsPanel 
              output={codeOutput} 
              error={executionError}
              isLoading={isExecuting}
              executionTime={executionTime}
            />
          </Split>
        </div>
      </Split>
    </div>
  );
}

export default PlaygroundLayout;