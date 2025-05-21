import React, { useState, useEffect } from 'react';
import Split from 'react-split';
import CodePanel from './CodePanel';
import AssistantPanel from './AssistantPanel';
import ResultsPanel from './ResultsPanel';
import QuestionPanel from './QuestionPanel';
import '../../styles/playground.css';

/**
 * PlaygroundLayout provides the main interface layout with resizable panels
 * for coding, getting assistance, viewing results, and reading questions.
 */
function PlaygroundLayout() {
  // State for panel sizes with default values
  const [horizontalSizes, setHorizontalSizes] = useState(() => {
    const saved = localStorage.getItem('horizontalSizes');
    return saved ? JSON.parse(saved) : [60, 40]; // Left/Right default split percentages
  });
  
  const [leftVerticalSizes, setLeftVerticalSizes] = useState(() => {
    const saved = localStorage.getItem('leftVerticalSizes');
    return saved ? JSON.parse(saved) : [30, 70]; // Question/Code default split percentages
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
    localStorage.setItem('leftVerticalSizes', JSON.stringify(leftVerticalSizes));
  }, [leftVerticalSizes]);

  useEffect(() => {
    localStorage.setItem('rightVerticalSizes', JSON.stringify(rightVerticalSizes));
  }, [rightVerticalSizes]);

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
    <div className="playground-container">
      <Split
        sizes={horizontalSizes}
        minSize={300}
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
            sizes={leftVerticalSizes}
            minSize={100}
            expandToMin={false}
            gutterSize={8}
            gutterAlign="center"
            direction="vertical"
            onDragEnd={setLeftVerticalSizes}
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
            minSize={100}
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
            />
          </Split>
        </div>
      </Split>
    </div>
  );
}

export default PlaygroundLayout;
