import React, { useState, useEffect } from 'react';
import Split from 'react-split';
import CodePanel from './CodePanel.tailwind.jsx';
import AssistantPanel from './AssistantPanel.tailwind.jsx';
import ResultsPanel from './ResultsPanel.tailwind.jsx';
import QuestionPanel from './QuestionPanel.tailwind.jsx';

/**
 * PlaygroundLayout provides the main interface layout with resizable panels
 * for coding, getting assistance, viewing results, and reading questions.
 * This version uses Tailwind CSS for styling.
 */
function PlaygroundLayout() {
  // State for panel sizes with default values
  const [horizontalSizes, setHorizontalSizes] = useState(() => {
    const saved = localStorage.getItem('horizontalSizes');
    return saved ? JSON.parse(saved) : [60, 40]; // Left/Right default split percentages
  });
  
  const [leftHorizontalSizes, setLeftHorizontalSizes] = useState(() => {
    const saved = localStorage.getItem('leftHorizontalSizes');
    return saved ? JSON.parse(saved) : [40, 60]; // Question/Code default split percentages
  });
  
  const [rightVerticalSizes, setRightVerticalSizes] = useState(() => {
    const saved = localStorage.getItem('rightVerticalSizes');
    return saved ? JSON.parse(saved) : [70, 30]; // Assistant/Results default split percentages
  });

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

  // Detect viewport width for responsive layout
  const [isSmallScreen, setIsSmallScreen] = useState(false);
  
  useEffect(() => {
    const checkScreenSize = () => {
      setIsSmallScreen(window.innerWidth < 768);
    };
    
    // Check on initial render
    checkScreenSize();
    
    // Add event listener for window resize
    window.addEventListener('resize', checkScreenSize);
    
    // Cleanup
    return () => window.removeEventListener('resize', checkScreenSize);
  }, []);

  // Handle code execution
  const [codeOutput, setCodeOutput] = useState('');
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionError, setExecutionError] = useState(null);

  const handleCodeExecution = async (code) => {
    setIsExecuting(true);
    setExecutionError(null);
    
    try {
      // This would be replaced with an actual API call
      const response = await simulateCodeExecution(code);
      setCodeOutput(response.output);
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

  return (
    <div className="flex flex-col h-screen w-full overflow-hidden bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100">
      <Split
        sizes={horizontalSizes}
        minSize={300}
        expandToMin={false}
        gutterSize={8}
        gutterAlign="center"
        direction="horizontal"
        className="flex h-full overflow-hidden"
        onDragEnd={setHorizontalSizes}
        gutterClassName="bg-slate-200 hover:bg-blue-500 transition-colors duration-200 dark:bg-slate-700 dark:hover:bg-blue-700"
      >
        {/* Left section: Question + Code Editor (horizontal layout) */}
        <div className="flex flex-col overflow-hidden">
          <Split
            sizes={leftHorizontalSizes}
            minSize={isSmallScreen ? 100 : 300}
            expandToMin={false}
            gutterSize={8}
            gutterAlign="center"
            direction={isSmallScreen ? "vertical" : "horizontal"}
            onDragEnd={setLeftHorizontalSizes}
            gutterClassName="bg-slate-200 hover:bg-blue-500 transition-colors duration-200 dark:bg-slate-700 dark:hover:bg-blue-700"
          >
            <QuestionPanel />
            <CodePanel 
              onRunCode={handleCodeExecution} 
              isExecuting={isExecuting} 
            />
          </Split>
        </div>

        {/* Right section: Assistant + Results */}
        <div className="flex flex-col overflow-hidden">
          <Split
            sizes={rightVerticalSizes}
            minSize={100}
            expandToMin={false}
            gutterSize={8}
            gutterAlign="center"
            direction="vertical"
            onDragEnd={setRightVerticalSizes}
            gutterClassName="bg-slate-200 hover:bg-blue-500 transition-colors duration-200 dark:bg-slate-700 dark:hover:bg-blue-700"
          >
            <AssistantPanel />
            <ResultsPanel 
              output={codeOutput} 
              error={executionError}
              isLoading={isExecuting}
            />
          </Split>
        </div>
      </Split>
    </div>
  );
}

export default PlaygroundLayout;